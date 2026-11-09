//! Compare one opening design against the same harness in the other worktree.
//! Run `RAYON_NUM_THREADS=6 cargo bench --bench shared_pair_compare -- 18 7`.

use std::{env::args, time::Instant};

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
        random_vector,
    },
    buffer::{Buffer, BufferMath, BufferOps},
    protocols::params::{
        FoldingFactor, Mode, PowBudget, ProtocolConfig, RateSchedule, SecuritySpec, TuningSpec,
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, ProverState, VerificationResult, VerifierMessage,
        VerifierState,
    },
};

const DESIGN: &str = "shared";

fn measure<M>(name: &str, log_size: u32, runs: usize)
where
    M: Embedding + Default,
    M::Target: Codec + Default,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let size = 1 << log_size;
    let config = ProtocolConfig::<M>::derive(
        SecuritySpec::new(128)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(PowBudget::Forbidden),
        TuningSpec {
            vector_size: 2 * size,
            starting_log_inv_rate: 2,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: 5,
                rest: 4,
            },
            rate_schedule: RateSchedule::Capped {
                max_log_inv_rate: 4,
            },
        },
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(731);
    let witnesses = [
        Buffer::random(&mut rng, size),
        Buffer::random(&mut rng, size),
    ];
    let point = random_vector(&mut rng, log_size as usize);
    let single_form = MultilinearExtension::new(point.clone());
    let pair_forms = [M::Target::ZERO, M::Target::ONE].map(|selector| {
        let mut full = vec![selector];
        full.extend_from_slice(&point);
        MultilinearExtension::new(full)
    });
    let forms: Vec<&dyn LinearForm<M::Target>> = pair_forms.iter().map(|f| f as _).collect();
    let values = witnesses
        .each_ref()
        .map(|w| single_form.evaluate(&M::default(), w.to_slice()));
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut timings = Vec::with_capacity(runs);
    for run in 0..=runs {
        let mut ps = ProverState::new_std(&ds);
        ps.prover_messages(&point);
        // Configuration derivation is excluded; API validation stays inside each timer.
        let started = Instant::now();
        let mut stage = M::Target::ZERO;
        let pair = config
            .commit_pair(
                &mut ps,
                witnesses[0].clone(),
                |ps| -> VerificationResult<_> {
                    stage = ps.verifier_message();
                    Ok(witnesses[1].clone())
                },
            )
            .unwrap();
        ps.prover_messages(&values);
        config.prove(&mut ps, pair, &forms, &values);
        let prove_ms = started.elapsed().as_secs_f64() * 1000.0;
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        assert_eq!(
            vs.prover_messages_vec::<M::Target>(point.len()).unwrap(),
            point
        );
        let started = Instant::now();
        let pair = config
            .receive_pair(&mut vs, |vs| -> VerificationResult<()> {
                assert_eq!(vs.verifier_message::<M::Target>(), stage);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            vs.prover_messages_vec::<M::Target>(values.len()).unwrap(),
            values
        );
        config
            .verify(&mut vs, pair, &forms, &values)
            .unwrap()
            .verify(&forms)
            .unwrap();
        vs.check_eof().unwrap();
        let verify_ms = started.elapsed().as_secs_f64() * 1000.0;
        if run != 0 {
            let bytes = proof.narg_string.len() + proof.hints.len();
            timings.push((prove_ms, verify_ms, bytes));
            println!("{name},2^{log_size},{DESIGN},{run},{prove_ms:.3},{verify_ms:.3},{bytes}");
        }
    }
    println!(
        "{name},2^{log_size},{DESIGN},median,{:.3},{:.3},{:.0}",
        median(timings.iter().map(|x| x.0).collect()),
        median(timings.iter().map(|x| x.1).collect()),
        median(timings.iter().map(|x| x.2 as f64).collect()),
    );
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    if args().any(|arg| arg == "--test") {
        return;
    }
    let log_size = args()
        .nth(1)
        .map_or(18, |v| v.parse().expect("log2 witness size"));
    let runs = args()
        .nth(2)
        .map_or(7, |v| v.parse().expect("measured runs"));
    assert!((12..=22).contains(&log_size) && runs > 0 && runs % 2 == 1);
    println!("field,witness_elements,design,sample,commit_prove_ms,receive_verify_ms,proof_bytes");
    measure::<Basefield<Field64_3>>("base_to_cubic", log_size, runs);
    measure::<Identity<Field64_3>>("cubic", log_size, runs);
    measure::<Identity<Field256>>("bn254", log_size, runs);
}
