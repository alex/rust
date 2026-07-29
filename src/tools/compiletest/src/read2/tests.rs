use std::io::Write;

use crate::read2::{FILTERED_PATHS_PLACEHOLDER_LEN, MAX_OUT_LEN, ProcOutput};

#[test]
fn test_abbreviate_short_string() {
    let mut out = ProcOutput::new();
    out.extend(b"Hello world!", &[]);
    assert_eq!(b"Hello world!", &*out.into_bytes());
}

#[test]
fn test_abbreviate_short_string_multiple_steps() {
    let mut out = ProcOutput::new();

    out.extend(b"Hello ", &[]);
    out.extend(b"world!", &[]);

    assert_eq!(b"Hello world!", &*out.into_bytes());
}

#[test]
fn test_abbreviate_long_string() {
    let mut out = ProcOutput::new();

    let data = vec![b'.'; MAX_OUT_LEN + 16];
    out.extend(&data, &[]);

    let mut expected = Vec::new();
    write!(expected, "<<<<<< TRUNCATED, SHOWING THE FIRST {MAX_OUT_LEN} BYTES >>>>>>\n\n").unwrap();
    expected.extend_from_slice(&[b'.'; MAX_OUT_LEN]);
    expected.extend_from_slice(b"\n\n<<<<<< TRUNCATED, DROPPED 16 BYTES >>>>>>");

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}

#[test]
fn test_abbreviate_filterss_are_detected() {
    let mut out = ProcOutput::new();
    // The filters have to be longer than the placeholder length, otherwise
    // discounting them could never bring the output back under the threshold.
    let filters = &["f".repeat(64), "q".repeat(64)];

    // The filtered length is only tracked once the raw output goes over the
    // truncation threshold, since below it the filters cannot change the
    // outcome. So make sure the output does go over it.
    out.extend(&vec![b'.'; MAX_OUT_LEN - 64], filters);
    out.extend(filters[0].as_bytes(), filters);
    // Check items are detected across extensions, and that items from a
    // previous extension are not double-counted.
    out.extend(&filters[1].as_bytes()[..32], filters);
    out.extend(&filters[1].as_bytes()[32..], filters);

    match &out {
        ProcOutput::Full { bytes, filtered_len } => assert_eq!(
            *filtered_len,
            Some(
                bytes.len() + FILTERED_PATHS_PLACEHOLDER_LEN * filters.len()
                    - filters.iter().map(|i| i.len()).sum::<usize>()
            )
        ),
        ProcOutput::Abbreviated { .. } => panic!("out should not be abbreviated"),
    }
}

#[test]
fn test_abbreviate_filters_avoid_abbreviations() {
    let mut out = ProcOutput::new();
    let filters = &["a".repeat(64)];

    let mut expected = vec![b'.'; MAX_OUT_LEN - FILTERED_PATHS_PLACEHOLDER_LEN];
    expected.extend_from_slice(filters[0].as_bytes());

    out.extend(&expected, filters);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}

#[test]
fn test_abbreviate_filters_can_still_cause_abbreviations() {
    let mut out = ProcOutput::new();
    let filters = &["a".repeat(64)];

    let mut input = vec![b'.'; MAX_OUT_LEN];
    input.extend_from_slice(filters[0].as_bytes());

    let mut expected = Vec::new();
    write!(expected, "<<<<<< TRUNCATED, SHOWING THE FIRST {MAX_OUT_LEN} BYTES >>>>>>\n\n").unwrap();
    expected.extend_from_slice(&[b'.'; MAX_OUT_LEN]);
    expected.extend_from_slice(b"\n\n<<<<<< TRUNCATED, DROPPED 64 BYTES >>>>>>");

    out.extend(&input, filters);

    // We first check the length to avoid endless terminal output if the length differs, since
    // `out` is hundreds of KBs in size.
    let out = out.into_bytes();
    assert_eq!(expected.len(), out.len());
    assert_eq!(expected, out);
}
