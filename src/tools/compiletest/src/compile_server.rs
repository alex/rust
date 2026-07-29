//! Client for `rustc`'s compile server.
//!
//! Running the test suite is dominated by process overhead: a `rustc`
//! invocation spends a fixed ~17ms on `execve`, dynamic loading, page faults
//! and teardown, which for `tests/ui` is paid roughly 24,000 times. The
//! compiler can instead stay resident and fork per compilation, so that each
//! one starts from a warm address space; see `compile_server` in
//! `rustc_driver_impl`.
//!
//! Servers are pooled rather than kept per thread, because compiletest runs
//! each test on a thread of its own and a per-thread server would be a fresh
//! process per test again.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, fs};

use camino::{Utf8Path, Utf8PathBuf};

/// Separates the fields of a request.
const FIELD: char = '\x01';
/// Separates the items within the environment and argument fields.
const ITEM: char = '\x02';

/// A resident `rustc` that forks per compilation.
#[derive(Debug)]
struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Where this server's children are told to write their output.
    scratch: Utf8PathBuf,
}

/// The result of a served compilation, as if it had been its own process.
pub(crate) struct Served {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

#[derive(Debug)]
pub(crate) struct ServerPool {
    rustc: Utf8PathBuf,
    /// Flags to warm a new server with, so that it loads the same standard
    /// library the tests will use.
    warmup_flags: Vec<String>,
    /// The target servers are warmed for; see [`Self::can_serve`].
    target: String,
    scratch_root: Utf8PathBuf,
    idle: Mutex<Vec<Server>>,
    next_id: AtomicUsize,
}

impl ServerPool {
    /// Creates a pool, unless serving is unavailable or has been turned off.
    ///
    /// Serving relies on `fork`, so it is Unix-only. `COMPILETEST_NO_COMPILE_SERVER`
    /// turns it off, which is worth reaching for if a test behaves differently
    /// under it: that would be a bug, but the escape hatch means it need not
    /// block anyone in the meantime.
    pub(crate) fn new(
        rustc: &Utf8Path,
        sysroot: &Utf8Path,
        target: &str,
        scratch_root: &Utf8Path,
    ) -> Option<Self> {
        if !cfg!(unix) || env::var_os("COMPILETEST_NO_COMPILE_SERVER").is_some() {
            return None;
        }
        let scratch_root = scratch_root.join(".compile-server");
        fs::create_dir_all(&scratch_root).ok()?;
        Some(Self {
            rustc: rustc.to_path_buf(),
            warmup_flags: vec![
                "--sysroot".to_owned(),
                sysroot.as_str().to_owned(),
                format!("--target={target}"),
            ],
            target: target.to_owned(),
            scratch_root,
            idle: Mutex::new(vec![]),
            next_id: AtomicUsize::new(0),
        })
    }

    fn checkout(&self) -> Server {
        if let Some(server) = self.idle.lock().unwrap().pop() {
            return server;
        }
        self.spawn()
    }

    fn checkin(&self, server: Server) {
        self.idle.lock().unwrap().push(server);
    }

    fn spawn(&self) -> Server {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let scratch = self.scratch_root.join(id.to_string());
        fs::create_dir_all(&scratch).expect("failed to create compile server scratch directory");

        let mut child = Command::new(&self.rustc)
            .env("RUSTC_COMPILE_SERVER", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to start compile server");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut server = Server { child, stdin, stdout, scratch };

        // Have the server compile something once, so that the heap it grew to
        // hold a session -- and the pages that took -- are resident and
        // inherited by every child instead of being rebuilt by each of them.
        // Use the flags the tests use, so the standard library gets loaded too.
        let src = server.scratch.join("warmup.rs");
        fs::write(&src, "pub fn warm() {}\n").expect("failed to write warmup source");
        let mut args = vec![self.rustc.as_str().to_owned()];
        args.extend(self.warmup_flags.iter().cloned());
        args.extend([
            "--crate-type=lib".to_owned(),
            "--emit=metadata".to_owned(),
            "--out-dir".to_owned(),
            server.scratch.as_str().to_owned(),
            src.as_str().to_owned(),
        ]);
        server.request("WARMUP", "", "", &[], &args);

        server
    }

    /// Whether `command` can be served, or whether it has to have a process of
    /// its own.
    ///
    /// A warmed server has already initialised things that the compiler only
    /// initialises once per process image and that depend on the compilation:
    /// the codegen backend is loaded, LLVM is set up for one target with one set
    /// of `-C llvm-args`, and `RUST_MIN_STACK` has been read. A child cannot
    /// redo any of that, so a compilation that would configure them differently
    /// is not interchangeable with the one the server was warmed with, and gets
    /// a process of its own.
    pub(crate) fn can_serve(&self, command: &Command) -> bool {
        let mut target = None;
        for arg in command.get_args() {
            let arg = arg.to_string_lossy();
            // Configures LLVM, or asks it something.
            if arg.starts_with("-Cllvm-args")
                || arg.starts_with("-Ctarget-cpu")
                || arg.starts_with("-Ctarget-feature")
                || arg.starts_with("-Zcodegen-backend")
                || arg.starts_with("--print")
            {
                return false;
            }
            if let Some(value) = arg.strip_prefix("--target") {
                target = Some(value.trim_start_matches('=').to_owned());
            }
        }
        // A compilation for some other target needs LLVM set up for it.
        if target.is_some_and(|target| target != self.target) {
            return false;
        }
        // Read once, on first use, and then cached by the standard library.
        !command
            .get_envs()
            .any(|(key, value)| key == "RUST_MIN_STACK" && value != Some("".as_ref()))
    }

    /// Runs `command` on a server, returning what it would have produced as its
    /// own process.
    pub(crate) fn run(&self, command: &Command) -> Served {
        let mut server = self.checkout();
        let served = server.run(command);
        self.checkin(server);
        served
    }
}

impl Server {
    fn request(
        &mut self,
        cwd: &str,
        stdout_path: &str,
        stderr_path: &str,
        env: &[String],
        args: &[String],
    ) -> String {
        let line = format!(
            "{cwd}{FIELD}{stdout_path}{FIELD}{stderr_path}{FIELD}{}{FIELD}{}\n",
            env.join(&ITEM.to_string()),
            args.join(&ITEM.to_string()),
        );
        self.stdin.write_all(line.as_bytes()).expect("failed to send compile server request");
        self.stdin.flush().expect("failed to flush compile server request");

        let mut reply = String::new();
        self.stdout.read_line(&mut reply).expect("failed to read compile server reply");
        reply.trim_end().to_owned()
    }

    fn run(&mut self, command: &Command) -> Served {
        let stdout_path = self.scratch.join("stdout");
        let stderr_path = self.scratch.join("stderr");

        // The child starts from an empty environment, so send the whole thing:
        // what this process would have passed on, with the command's own
        // overrides applied and its removals honoured.
        let mut env: Vec<(String, String)> = env::vars().collect();
        for (key, value) in command.get_envs() {
            let key = key.to_string_lossy().into_owned();
            env.retain(|(k, _)| *k != key);
            if let Some(value) = value {
                env.push((key, value.to_string_lossy().into_owned()));
            }
        }
        let env: Vec<String> = env.into_iter().map(|(k, v)| format!("{k}={v}")).collect();

        let mut args = vec![command.get_program().to_string_lossy().into_owned()];
        args.extend(command.get_args().map(|a| a.to_string_lossy().into_owned()));

        let cwd = command
            .get_current_dir()
            .map(|d| d.to_string_lossy().into_owned())
            .unwrap_or_else(|| env::current_dir().unwrap().to_string_lossy().into_owned());

        let reply = self.request(&cwd, stdout_path.as_str(), stderr_path.as_str(), &env, &args);

        let status = parse_status(&reply);
        let stdout = fs::read(&stdout_path).unwrap_or_default();
        let stderr = fs::read(&stderr_path).unwrap_or_default();
        Served { status, stdout, stderr }
    }
}

/// Turns a `##EXIT`/`##SIGNAL` reply back into the [`ExitStatus`] the caller
/// would have seen from a real process, so that tests asserting a particular
/// exit code (or a crash) behave identically.
#[cfg(unix)]
fn parse_status(reply: &str) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;

    if let Some(code) = reply.strip_prefix("##EXIT ") {
        let code: i32 = code.parse().expect("compile server reported a bad exit code");
        ExitStatus::from_raw(code << 8)
    } else if let Some(signal) = reply.strip_prefix("##SIGNAL ") {
        let signal: i32 = signal.parse().expect("compile server reported a bad signal");
        ExitStatus::from_raw(signal)
    } else {
        panic!("unexpected compile server reply: {reply:?}")
    }
}

#[cfg(not(unix))]
fn parse_status(_reply: &str) -> ExitStatus {
    // Unreachable: `ServerPool::new` returns `None` off Unix.
    unreachable!("the compile server is Unix-only")
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
