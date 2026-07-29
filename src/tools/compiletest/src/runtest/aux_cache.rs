//! Sharing of auxiliary crate builds between tests.
//!
//! Auxiliaries are compiled from their own directives plus the revision of the
//! test that asked for them, so the same auxiliary requested by two different
//! tests almost always produces byte-identical artifacts. Some auxiliaries are
//! popular enough for that to matter: `tests/ui/proc-macro/auxiliary/
//! test-macros.rs` alone is asked for by 80-odd tests, and building a
//! proc-macro means a full codegen and link each time.
//!
//! So each distinct auxiliary build is performed once, into a directory of its
//! own, and tests that need it get the artifacts hard-linked into their private
//! auxiliary directory. Keeping the per-test directories means the `-L` path
//! handed to `rustc` still contains only the auxiliaries that test asked for,
//! so nothing about crate resolution changes.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use std::{fs, io};

use build_helper::fs::remove_and_create_dir_all;
use camino::{Utf8Path, Utf8PathBuf};

use crate::runtest::{AuxType, ProcRes};

/// Identifies an auxiliary build that can be shared between tests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AuxKey {
    /// Path of the auxiliary's source file.
    pub(crate) source: Utf8PathBuf,
    /// Revision of the test that asked for the build. Auxiliaries are compiled
    /// with the requesting test's `--cfg <revision>`, so builds requested by
    /// different revisions are not interchangeable.
    pub(crate) revision: Option<String>,
    /// Aux type demanded by the directive that asked for the build, if any.
    pub(crate) forced_type: Option<AuxType>,
}

impl AuxKey {
    /// Directory that this auxiliary's artifacts live in, below `suite_root`.
    ///
    /// The stem is only there to make the build directory easier to read; the
    /// hash is what keeps distinct auxiliaries apart.
    fn dir(&self, suite_root: &Utf8Path) -> Utf8PathBuf {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let stem = self.source.file_stem().unwrap_or("aux");
        // Test output directories are named after test files, which never start
        // with a `.`, so this cannot collide with one.
        suite_root.join(".aux-cache").join(format!("{stem}-{:016x}", hasher.finish()))
    }
}

/// A completed auxiliary build, kept around so that other tests needing the
/// same auxiliary can link in its artifacts instead of running `rustc` again.
#[derive(Debug)]
pub(crate) struct AuxBuild {
    /// Directory that the artifacts were built into.
    dir: Utf8PathBuf,
    /// Artifact paths, relative to [`Self::dir`].
    files: Vec<Utf8PathBuf>,
    /// The aux type that the build settled on.
    pub(crate) aux_type: AuxType,
    /// The failed `rustc` invocation, if the build did not succeed. Reported
    /// again for every test that asks for this auxiliary, so that a broken
    /// auxiliary fails all of its dependents and not just the first one.
    pub(crate) failure: Option<ProcRes>,
}

impl AuxBuild {
    fn new(dir: Utf8PathBuf, aux_type: AuxType, res: ProcRes) -> Self {
        let files =
            if res.status.success() { collect_files(&dir, Utf8Path::new("")) } else { vec![] };
        let failure = if res.status.success() { None } else { Some(res) };
        Self { dir, files, aux_type, failure }
    }

    /// Makes this build's artifacts available in `dest`, as if they had been
    /// compiled there.
    ///
    /// Hard links are used so that this stays cheap and doesn't eat disk space;
    /// nothing ever writes to the artifacts after they are built, so sharing
    /// the underlying files is safe.
    pub(crate) fn link_into(&self, dest: &Utf8Path) -> io::Result<()> {
        for file in &self.files {
            let from = self.dir.join(file);
            let to = dest.join(file);
            if let Some(parent) = to.parent() {
                fs::create_dir_all(parent)?;
            }
            // `hard_link` fails if the destination exists, and a previous run
            // may well have left something behind.
            match fs::remove_file(&to) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
            // Hard links can't cross filesystems, and some filesystems don't
            // support them at all, so fall back to copying.
            if fs::hard_link(&from, &to).is_err() {
                fs::copy(&from, &to)?;
            }
        }
        Ok(())
    }
}

/// Auxiliary builds shared by all the tests in a run.
#[derive(Debug, Default)]
pub(crate) struct AuxCache {
    entries: Mutex<HashMap<AuxKey, Arc<OnceLock<AuxBuild>>>>,
}

impl AuxCache {
    /// Returns the shared build for `key`, performing it with `build` if this
    /// is the first time it has been asked for.
    ///
    /// `build` is handed the (freshly emptied) directory to build into. It runs
    /// at most once per key; callers that ask for the same key concurrently
    /// wait for the in-progress build instead of starting their own.
    pub(crate) fn get_or_build(
        &self,
        suite_root: &Utf8Path,
        key: AuxKey,
        build: impl FnOnce(&Utf8Path) -> (AuxType, ProcRes),
    ) -> Arc<OnceLock<AuxBuild>> {
        // Only the map is locked here, not the build itself, so that unrelated
        // auxiliaries can still be built in parallel.
        let cell = {
            let mut entries = self.entries.lock().unwrap();
            Arc::clone(entries.entry(key.clone()).or_default())
        };

        cell.get_or_init(|| {
            let dir = key.dir(suite_root);
            remove_and_create_dir_all(&dir).unwrap_or_else(|e| {
                panic!("failed to remove and recreate auxiliary directory `{dir}`: {e}")
            });
            let (aux_type, res) = build(&dir);
            AuxBuild::new(dir, aux_type, res)
        });

        cell
    }
}

/// Lists every file below `dir`, as paths relative to it.
fn collect_files(dir: &Utf8Path, relative_to_dir: &Utf8Path) -> Vec<Utf8PathBuf> {
    let mut files = vec![];
    let Ok(entries) = fs::read_dir(dir.join(relative_to_dir).as_std_path()) else {
        return files;
    };
    for entry in entries.flatten() {
        let Ok(name) = Utf8PathBuf::try_from(std::path::PathBuf::from(entry.file_name())) else {
            continue;
        };
        let relative = relative_to_dir.join(name);
        match entry.file_type() {
            Ok(ty) if ty.is_dir() => files.extend(collect_files(dir, &relative)),
            Ok(_) => files.push(relative),
            Err(_) => {}
        }
    }
    files
}
