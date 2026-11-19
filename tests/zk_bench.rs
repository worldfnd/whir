//! Zook commit/prove/verify timing (Goldilocks witness, cubic extension by default).
//!
//! ```text
//! LOG=20 MODE=std ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
//! ```
//!
//! Settings come from environment variables; see the README section "Benchmarking Zook".
//! Prints the best time over `ITERS` runs per phase, plus a digest of the Standard-mode proof.

use std::time::Instant;

use ark_std::rand::{
    distributions::{Distribution, Standard},
    rngs::StdRng,
    SeedableRng,
};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::Field64_3,
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
        random_vector,
    },
    buffer::Buffer,
    hash,
    protocols::params::{
        DecodingRegime, FoldingFactor, Mode, PowBudget, ProtocolConfig, RateSchedule, SecuritySpec,
        TuningSpec,
    },
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

fn env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn parse<T: std::str::FromStr>(key: &str, default: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    env(key, default)
        .parse()
        .unwrap_or_else(|e| panic!("invalid {key}: {e:?}"))
}

/// FNV-1a, enough to tell whether two runs produced the same proof bytes.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn run<M>(log_size: u32, mode: Mode, iterations: usize)
where
    M: Embedding + Default,
    M::Target: Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let size = 1usize << log_size;
    let spec = SecuritySpec {
        mode,
        decoding_regime: DecodingRegime::Johnson,
        target_security_bits: parse("SEC", "128"),
        pow_budget: PowBudget::per_slot(parse("POW", "10")),
        hash_id: match env("HASH", "sha2").as_str() {
            "blake3" => hash::BLAKE3,
            "sha2" => hash::SHA2,
            other => panic!("invalid HASH: {other}"),
        },
    };
    let tuning = TuningSpec {
        vector_size: size,
        starting_log_inv_rate: parse("RATE", "2"),
        folding_factor: FoldingFactor::Constant(parse("FOLD", "4")),
        rate_schedule: RateSchedule::Stepping,
    };
    let config = ProtocolConfig::<M>::derive(spec, tuning)
        .unwrap_or_else(|e| panic!("config for 2^{log_size} {mode:?}: {e:?}"));

    let mut rng = StdRng::seed_from_u64(0);
    let witness: Vec<M::Source> = random_vector(&mut rng, size);
    let form = MultilinearExtension {
        point: random_vector::<M::Target>(&mut rng, log_size as usize),
    };
    let value = form.evaluate(&M::default(), &witness);
    let form: &dyn LinearForm<M::Target> = &form;

    let (mut commit, mut prove, mut verify) = (f64::MAX, f64::MAX, f64::MAX);
    let mut proof_bytes = Vec::new();
    for _ in 0..iterations {
        let ds = DomainSeparator::protocol(&"zook-bench").instance(&Empty);
        let mut ps = ProverState::new_std(&ds);

        let start = Instant::now();
        let committed = config.commit(&mut ps, Buffer::from(witness.clone()));
        commit = commit.min(start.elapsed().as_secs_f64());

        let start = Instant::now();
        let _ = config.prove(&mut ps, committed, &[form], &[value]);
        prove = prove.min(start.elapsed().as_secs_f64());

        let proof = ps.proof();
        proof_bytes.clone_from(&proof.narg_string);

        let mut vs = VerifierState::new_std(&ds, &proof);
        let start = Instant::now();
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let claim = config
            .verify(&mut vs, commitment, &[form], &[value])
            .unwrap();
        claim.verify(&[form]).unwrap();
        verify = verify.min(start.elapsed().as_secs_f64());
    }

    // ZK proofs are randomised, so only the Standard-mode digest is comparable across runs.
    let proof_digest = match mode {
        Mode::Standard => format!("{:016x}", digest(&proof_bytes)),
        Mode::ZeroKnowledge => "-".to_string(),
    };
    println!(
        "RESULT log={log_size} mode={mode:?} commit={commit:.3}s prove={prove:.3}s \
         verify={verify:.4}s proof_digest={proof_digest}"
    );
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn zook_bench() {
    let log_size: u32 = parse("LOG", "20");
    let iterations: usize = parse("ITERS", "3");
    for mode in env("MODE", "std,zk").split(',') {
        let mode = match mode {
            "std" => Mode::Standard,
            "zk" => Mode::ZeroKnowledge,
            other => panic!("invalid MODE: {other}"),
        };
        match env("EMB", "base").as_str() {
            "base" => run::<Basefield<Field64_3>>(log_size, mode, iterations),
            "ext" => run::<Identity<Field64_3>>(log_size, mode, iterations),
            other => panic!("invalid EMB: {other}"),
        }
    }
}
