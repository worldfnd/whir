//! zkWHIR benchmark keyed to ProveKit v1 parameters.
//!
//! Source of truth for params:
//! `provekit/provekit/r1cs-compiler/src/whir_r1cs.rs:75` (v1 branch).
//!
//! Run with:
//!   cargo bench --bench zk_whir
//!   cargo bench --bench zk_whir -- commit
//!   cargo bench --bench zk_whir -- prove
//!   cargo bench --bench zk_whir -- verify

use std::borrow::Cow;

use ark_ff::FftField;
use ark_std::rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng};
use divan::{black_box, AllocProfiler, Bencher};
use whir::{
    algebra::{
        embedding::Embedding,
        fields::Field256,
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
        MultilinearPoint,
    },
    hash,
    parameters::ProtocolParameters,
    protocols::whir_zk,
    transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

type F = Field256;

/// num_variables (μ) sweep. Anchors at real ProveKit circuit sizes:
///   14 = MIN_WHIR_NUM_VARIABLES floor (basic-4, poseidon2).
///   19 = poseidon-rounds — ProveKit's only `provekit-bench` circuit.
///   20 = complete_age_check — passport hot circuit (HOTSPOTS reference).
const SIZES: &[usize] = &[14, 19, 20];

/// ProveKit commits one polynomial at a time on the prove path.
const NUM_POLYS: usize = 1;

/// ProveKit zkWHIR parameters (provekit/r1cs-compiler/src/whir_r1cs.rs:75, v1).
fn provekit_params() -> ProtocolParameters {
    ProtocolParameters {
        unique_decoding:        false,
        security_level:         128,
        pow_bits:               10,
        initial_folding_factor: 3,
        folding_factor:         3,
        starting_log_inv_rate:  2,
        batch_size:             1,
        hash_id:                hash::SHA2,
    }
}

fn make_config(num_variables: usize) -> whir_zk::Config<F> {
    whir_zk::Config::new(num_variables, &provekit_params())
}

fn make_polynomials(num_variables: usize, num_polynomials: usize) -> Vec<Vec<F>> {
    let num_coeffs = 1usize << num_variables;
    (0..num_polynomials)
        .map(|poly_idx| {
            (0..num_coeffs)
                .map(|coeff_idx| F::from((poly_idx * num_coeffs + coeff_idx + 1) as u64))
                .collect()
        })
        .collect()
}

fn make_weights_and_evaluations<E: Embedding<Source = F, Target = F>>(
    polynomials: &[Vec<F>],
    embedding: &E,
    num_variables: usize,
) -> (Vec<MultilinearExtension<F>>, Vec<F>)
where
    Standard: Distribution<F>,
    F: FftField,
{
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let point = MultilinearPoint::<F>::rand(&mut rng, num_variables);
    let linear_form = MultilinearExtension::new(point.0);
    let mut evaluations = Vec::with_capacity(polynomials.len());
    for poly in polynomials {
        evaluations.push(linear_form.evaluate(embedding, poly));
    }
    (vec![linear_form], evaluations)
}

#[divan::bench(args = SIZES)]
fn commit(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let config = make_config(num_variables);
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-commit-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| ProverState::new_std(&ds))
        .bench_values(|mut prover_state| {
            let poly_refs: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
            let _ = black_box(config.commit(&mut prover_state, &poly_refs));
        });
}

#[divan::bench(args = SIZES)]
fn prove(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let config = make_config(num_variables);
    let (weights, evaluations) = make_weights_and_evaluations(
        &polynomials,
        config.blinded_polynomial.embedding(),
        num_variables,
    );
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-prove-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let poly_refs: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
            let witness = config.commit(&mut prover_state, &poly_refs);
            (prover_state, witness)
        })
        .bench_values(|(mut prover_state, witness)| {
            let poly_refs: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
            let prove_forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(
                MultilinearExtension::new(weights[0].point.clone()),
            )];
            black_box(config.prove(
                &mut prover_state,
                poly_refs
                    .iter()
                    .map(|v| Cow::Borrowed(*v))
                    .collect::<Vec<_>>(),
                witness,
                prove_forms,
                Cow::Borrowed(evaluations.as_slice()),
            ));
        });
}

#[divan::bench(args = SIZES)]
fn verify(bencher: Bencher, num_variables: usize) {
    let polynomials = make_polynomials(num_variables, NUM_POLYS);
    let config = make_config(num_variables);
    let (weights, evaluations) = make_weights_and_evaluations(
        &polynomials,
        config.blinded_polynomial.embedding(),
        num_variables,
    );
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-verify-{num_variables}"))
        .instance(&Empty);

    let proof = {
        let mut prover_state = ProverState::new_std(&ds);
        let poly_refs: Vec<&[F]> = polynomials.iter().map(Vec::as_slice).collect();
        let witness = config.commit(&mut prover_state, &poly_refs);
        let prove_forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(MultilinearExtension::new(
            weights[0].point.clone(),
        ))];
        let _ = config.prove(
            &mut prover_state,
            poly_refs
                .iter()
                .map(|v| Cow::Borrowed(*v))
                .collect::<Vec<_>>(),
            witness,
            prove_forms,
            Cow::Borrowed(evaluations.as_slice()),
        );
        prover_state.proof()
    };

    bencher
        .with_inputs(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitments = config.receive_commitments(&mut verifier_state).unwrap();
            (verifier_state, commitments)
        })
        .bench_values(|(mut verifier_state, commitments)| {
            let weight_refs = [&weights[0] as &dyn LinearForm<F>];
            let _ = config
                .verify(&mut verifier_state, &weight_refs, &evaluations, &commitments)
                .unwrap();
            black_box(());
        });
}

#[cfg(feature = "tracing")]
fn install_flame_subscriber() -> Option<tracing_flame::FlushGuard<std::io::BufWriter<std::fs::File>>> {
    use tracing_subscriber::{layer::SubscriberExt, registry::Registry, util::SubscriberInitExt};
    let path = std::env::var("WHIR_FLAME_PATH").ok()?;
    let (flame_layer, guard) = tracing_flame::FlameLayer::with_file(&path).ok()?;
    Registry::default().with(flame_layer).init();
    eprintln!("tracing-flame: writing folded stacks to {path}");
    Some(guard)
}

fn main() {
    #[cfg(feature = "tracing")]
    let _flame_guard = install_flame_subscriber();
    divan::main();
}
