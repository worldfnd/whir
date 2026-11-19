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
    protocols::params::{JointOpeningConfig, Mode, Params, ProtocolConfig, RateSchedule},
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

fn measure<M>(name: &str, log_size: u32, runs: usize)
where
    M: Embedding + Default,
    M::Target: Codec + Default,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let size = 1 << log_size;
    let params = Params {
        bits: 128,
        mode: Mode::ZeroKnowledge,
        regime: whir::protocols::params::DecodingRegime::Johnson,
        pow: 0,
        hash: whir::hash::BLAKE3,
        size,
        rate: 2,
        folds: [4, 4],
        schedule: RateSchedule::Capped(4),
    };
    let separate = ProtocolConfig::<M>::derive(params).unwrap();
    let joint = JointOpeningConfig::<M>::derive(params).unwrap();
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
    let domains = [
        DomainSeparator::protocol(&joint).instance(&Empty),
        DomainSeparator::protocol(&separate).instance(&Empty),
    ];
    for run in 0..=runs {
        // Alternate order and exclude one warmup for each path.
        for method in if run % 2 == 0 { [0, 1] } else { [1, 0] } {
            let ds = &domains[method];
            let mut ps = ProverState::new_std(ds);
            ps.prover_messages(&point);
            let started = Instant::now();
            if method == 0 {
                joint
                    .commit_first(&mut ps, witnesses[0].clone())
                    .commit_second(&mut ps, witnesses[1].clone())
                    .prove(&mut ps, &pair_refs, &values);
            } else {
                let handles = witnesses
                    .each_ref()
                    .map(|w| separate.commit(&mut ps, w.clone()));
                for (handle, &value) in handles.into_iter().zip(&values) {
                    separate.prove(&mut ps, handle, &[&single_form], &[value]);
                }
            }
            let prove_ms = started.elapsed().as_secs_f64() * 1000.0;
            let proof = ps.proof();
            let mut vs = VerifierState::new_std(ds, &proof);
            assert_eq!(
                vs.prover_messages_vec::<M::Target>(point.len()).unwrap(),
                point
            );
            let started = Instant::now();
            if method == 0 {
                joint
                    .receive_first(&mut vs)
                    .unwrap()
                    .receive_second(&mut vs)
                    .unwrap()
                    .verify(&mut vs, &pair_refs, &values)
                    .unwrap()
                    .verify(&pair_refs)
                    .unwrap();
            } else {
                let handles = [
                    separate.receive_commitment(&mut vs).unwrap(),
                    separate.receive_commitment(&mut vs).unwrap(),
                ];
                for (handle, &value) in handles.into_iter().zip(&values) {
                    separate
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
    print_timings(name, log_size, &timings);
}

fn print_timings(name: &str, log_size: u32, timings: &[Vec<(f64, f64, usize)>; 2]) {
    for (method, samples) in ["pair", "separate"].into_iter().zip(timings) {
        println!(
            "{name},2^{log_size},{method},{:.3},{:.3},{:.1}",
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
