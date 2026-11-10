//! Compare complete pair openings with two independent openings of the same claims.
//! Run `RAYON_NUM_THREADS=6 cargo bench --bench pair_opening -- 18 5`.

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
    protocols::params::{Mode, ProtocolConfig, RateSchedule, SecuritySpec, TuningSpec},
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

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
            .with_pow_budget(0),
        TuningSpec {
            vector_size: size,
            starting_log_inv_rate: 2,
            folding_factor: [4, 4],
            rate_schedule: RateSchedule::Capped {
                max_log_inv_rate: 4,
            },
        },
    )
    .unwrap();
    config.validate_pair().unwrap();
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
    let pair_refs: Vec<&dyn LinearForm<M::Target>> = pair_forms.iter().map(|f| f as _).collect();
    let values = witnesses
        .each_ref()
        .map(|w| single_form.evaluate(&M::default(), w.to_slice()));
    let mut timings = [Vec::new(), Vec::new()];
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    for run in 0..=runs {
        // Alternate order and exclude one warmup for each path.
        for method in if run % 2 == 0 { [0, 1] } else { [1, 0] } {
            let mut ps = ProverState::new_std(&ds);
            ps.prover_messages(&point);
            let started = Instant::now();
            let handles = witnesses
                .each_ref()
                .map(|w| config.commit(&mut ps, w.clone()));
            if method == 0 {
                config
                    .prove_pair(&mut ps, handles, &pair_refs, &values)
                    .unwrap();
            } else {
                for (handle, &value) in handles.into_iter().zip(&values) {
                    config.prove(&mut ps, handle, &[&single_form], &[value]);
                }
            }
            let prove_ms = started.elapsed().as_secs_f64() * 1000.0;
            let proof = ps.proof();
            let mut vs = VerifierState::new_std(&ds, &proof);
            assert_eq!(
                vs.prover_messages_vec::<M::Target>(point.len()).unwrap(),
                point
            );
            let started = Instant::now();
            let handles = [
                config.receive_commitment(&mut vs).unwrap(),
                config.receive_commitment(&mut vs).unwrap(),
            ];
            if method == 0 {
                config
                    .verify_pair(&mut vs, handles, &pair_refs, &values)
                    .unwrap()
                    .verify(&pair_refs)
                    .unwrap();
            } else {
                for (handle, &value) in handles.into_iter().zip(&values) {
                    config
                        .verify(&mut vs, handle, &[&single_form], &[value])
                        .unwrap()
                        .verify(&[&single_form])
                        .unwrap();
                }
            }
            vs.check_eof().unwrap();
            let verify_ms = started.elapsed().as_secs_f64() * 1000.0;
            if run != 0 {
                timings[method].push((
                    prove_ms,
                    verify_ms,
                    proof.narg_string.len() + proof.hints.len(),
                ));
            }
        }
    }
    for (method, samples) in timings.into_iter().enumerate() {
        println!(
            "{name},2^{log_size},{},{:.3},{:.3},{:.1}",
            if method == 0 { "pair" } else { "separate" },
            median(samples.iter().map(|x| x.0).collect()),
            median(samples.iter().map(|x| x.1).collect()),
            median(samples.iter().map(|x| x.2 as f64).collect()) / 1024.0,
        );
    }
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
        .map_or(5, |v| v.parse().expect("measured runs"));
    assert!((12..=22).contains(&log_size) && runs > 0 && runs % 2 == 1);
    println!("field,witness_elements,opening,commit_prove_ms,receive_verify_ms,proof_kib");
    measure::<Basefield<Field64_3>>("base_to_cubic", log_size, runs);
    measure::<Identity<Field64_3>>("cubic", log_size, runs);
    measure::<Identity<Field256>>("bn254", log_size, runs);
}
