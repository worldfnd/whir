//! Guard the two witness-sized transfer seams fixed by the resident prover path.
//!
//! This deliberately checks the protocol boundary rather than CPU backend
//! internals: CPU encoders may use slices privately, while resident protocol
//! code must not materialize large host vectors.

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).expect("start marker");
    let tail = &source[start..];
    let end = tail.find(end).expect("end marker");
    &tail[..end]
}

#[test]
fn resident_hot_path_has_no_full_host_bounces() {
    let irs = include_str!("../src/protocols/irs_commit.rs");
    let irs_commit = section(
        irs,
        "    pub fn commit<H, R>(",
        "    /// Receive a commitment",
    );
    assert!(!irs_commit.contains(".to_slice()"));
    assert!(!irs_commit.contains(".into_vec()"));
    assert!(!irs_commit.contains("poly_buf"));

    let zook = include_str!("../src/protocols/zook/prover.rs");
    let prove_round = section(
        zook,
        "fn prove_round<M, H, R>(",
        "/// The committed mask tree",
    );
    assert!(!prove_round.contains(".into_vec()"));
    assert!(!prove_round.contains("Buffer::from(message)"));
    assert!(!prove_round.contains("Buffer::from(covector)"));

    let code_switch = include_str!("../src/protocols/code_switch.rs");
    let prove = section(
        code_switch,
        "    pub fn prove<H, R>(",
        "    /// Send OOD answers",
    );
    assert!(!prove.contains("Buffer::from(message"));
    assert!(!prove.contains("message.into_vec()"));
    assert!(!prove.contains("covector.into_vec()"));
    assert!(!prove.contains("Buffer::from(ood_answers)"));
    assert!(!prove.contains(".to_slice()"));
}
