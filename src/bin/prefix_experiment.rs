use std::{
    borrow::Cow,
    iter::once,
    time::{Duration, Instant},
};

use ark_ff::{AdditiveGroup, PrimeField};
use clap::{Parser, ValueEnum};
#[cfg(feature = "parallel")]
use rayon::current_num_threads;
use serde::Serialize;
use serde_json::to_string;
use whir::{
    algebra::{
        embedding::Identity,
        fields::Field256,
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
    },
    hash::{HASH_COUNTER, SHA2},
    parameters::ProtocolParameters,
    protocols::{
        matrix_commit::Encodable,
        whir::Config as WhirConfig,
        whir_prefix::{Config as PrefixConfig, Dimensions, TensorLinearForm},
        whir_zk::Config as LegacyConfig,
    },
    transcript::{
        codecs::Empty, DomainSeparator, Proof, ProverState, VerificationResult, VerifierMessage,
        VerifierState,
    },
    verify,
};

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
enum Mode {
    Plain,
    LegacyHiding,
    Prefix,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum, default_value_t = Mode::Prefix)]
    mode: Mode,
    #[arg(long, default_value_t = 14)]
    variables: usize,
    /// Nonzero payload length before caller padding; defaults to 2^variables.
    #[arg(long)]
    witness_size: Option<usize>,
    #[arg(long, default_value_t = 5)]
    folds: usize,
    #[arg(long, default_value_t = 148)]
    queries: usize,
    #[arg(long, default_value_t = 1)]
    ood: usize,
    #[arg(long, default_value_t = 2)]
    log_inv_rate: usize,
    #[arg(long, default_value_t = 2048)]
    base_size: usize,
    #[arg(long, default_value_t = 128)]
    security_bits: usize,
    /// Use v1's unique-decoding parameters; supported only in plain mode.
    #[arg(long)]
    unique_decoding: bool,
    #[arg(long, default_value_t = 20)]
    verifier_repetitions: u32,
    #[arg(long, default_value_t = 3)]
    runs: usize,
}

#[derive(Serialize)]
struct Measurement {
    mode: Mode,
    variables: usize,
    witness_size: usize,
    run: usize,
    field: &'static str,
    field_modulus: String,
    hash: &'static str,
    security_target_bits: usize,
    baseline_unique_decoding: Option<bool>,
    candidate_security_certified: bool,
    verifier_repetitions: u32,
    folds: usize,
    queries: Option<usize>,
    ood: Option<usize>,
    log_inv_rate: usize,
    rayon_threads: usize,
    oracles: Vec<OracleParameters>,
    oracle_metadata_scope: &'static str,
    commitment_ms: f64,
    claim_evaluation_ms: f64,
    opening_ms: f64,
    verification_ms: f64,
    proof_bytes: usize,
    transcript_bytes: usize,
    hint_bytes: usize,
    prover_hashes: usize,
    verifier_hashes: f64,
}

#[derive(Serialize)]
struct OracleParameters {
    name: String,
    columns: usize,
    coefficient_count: usize,
    column_coefficients: usize,
    rows: usize,
    configured_in_domain_draws: usize,
    ood_answers: usize,
    prefix_per_column: usize,
    rate: f64,
    leaf_bytes: usize,
}

fn baseline_oracles(label: &str, config: &WhirConfig<Identity<Field256>>) -> Vec<OracleParameters> {
    once(&config.initial_committer)
        .chain(
            config
                .round_configs
                .iter()
                .map(|round| &round.irs_committer),
        )
        .enumerate()
        .map(|(index, oracle)| OracleParameters {
            name: format!("{label}.{index}"),
            columns: oracle.num_vectors * oracle.interleaving_depth,
            coefficient_count: oracle.vector_size * oracle.num_vectors,
            column_coefficients: oracle.message_length(),
            rows: oracle.codeword_length,
            configured_in_domain_draws: oracle.in_domain_samples,
            ood_answers: oracle.out_domain_samples * oracle.num_vectors,
            prefix_per_column: 0,
            rate: oracle.rate(),
            leaf_bytes: oracle.matrix_commit.num_cols * Field256::encoded_size(),
        })
        .collect()
}

fn prefix_oracles(config: &PrefixConfig<Field256>, args: &Args) -> Vec<OracleParameters> {
    let record = |name: String, dimensions: Dimensions, ood, prefix| OracleParameters {
        name,
        columns: dimensions.columns,
        coefficient_count: dimensions.columns * dimensions.column_size,
        column_coefficients: dimensions.column_size,
        rows: dimensions.rows,
        configured_in_domain_draws: args.queries,
        ood_answers: ood,
        prefix_per_column: prefix,
        rate: dimensions.column_size as f64 / dimensions.rows as f64,
        leaf_bytes: dimensions.row_width::<Field256>() * Field256::encoded_size(),
    };
    let mut oracles = Vec::new();
    for (index, &dimensions) in config.dimensions().iter().enumerate() {
        oracles.push(record(
            format!("main.{index}"),
            dimensions,
            args.ood,
            dimensions.prefix,
        ));
        if index + 1 < config.dimensions().len() {
            let mask = config.mask_dimensions();
            oracles.push(record(
                format!("sumcheck_mask.{index}"),
                mask,
                0,
                mask.prefix,
            ));
        } else {
            oracles.push(record("final_full_mask".to_owned(), dimensions, 0, 0));
        }
    }
    oracles
}

#[derive(Clone, Copy)]
struct Timings {
    commitment: Duration,
    evaluation: Duration,
    opening: Duration,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if args.variables >= usize::BITS as usize || args.runs == 0 || args.verifier_repetitions == 0 {
        return Err("variables must fit usize; runs and repetitions must be positive".to_owned());
    }
    let witness_size = args.witness_size.unwrap_or(1usize << args.variables);
    if witness_size == 0
        || witness_size.checked_next_power_of_two() != Some(1usize << args.variables)
    {
        return Err("witness size must round up to 2^variables".to_owned());
    }
    if args.unique_decoding && !matches!(args.mode, Mode::Plain) {
        return Err("--unique-decoding is supported only in plain mode".to_owned());
    }
    let parameters = ProtocolParameters {
        unique_decoding: args.unique_decoding,
        starting_log_inv_rate: args.log_inv_rate,
        initial_folding_factor: args.folds,
        folding_factor: args.folds,
        security_level: args.security_bits,
        pow_bits: 0,
        batch_size: 1,
        hash_id: SHA2,
    };
    let prefix_config = if matches!(args.mode, Mode::Prefix) {
        Some(
            PrefixConfig::new(
                witness_size,
                args.folds,
                args.queries,
                args.ood,
                args.log_inv_rate,
                args.base_size,
                SHA2,
            )
            .map_err(str::to_owned)?,
        )
    } else {
        None
    };
    for run in 0..args.runs {
        let vector: Vec<_> = (0..1usize << args.variables)
            .map(|i| {
                if i < witness_size {
                    Field256::from(i as u64 + 1)
                } else {
                    Field256::ZERO
                }
            })
            .collect();
        let result = match args.mode {
            Mode::Plain => plain(&args, &parameters, &vector),
            Mode::LegacyHiding => legacy(&args, &parameters, &vector),
            Mode::Prefix => prefix(
                &args,
                prefix_config
                    .as_ref()
                    .ok_or("missing prefix configuration")?,
                &vector,
            ),
        };
        let mut measurement =
            result.map_err(|_| "proof generation or verification failed".to_owned())?;
        measurement.run = run;
        println!("{}", to_string(&measurement).map_err(|e| e.to_string())?);
    }
    Ok(())
}

fn prefix(
    args: &Args,
    config: &PrefixConfig<Field256>,
    vector: &[Field256],
) -> VerificationResult<Measurement> {
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut state = ProverState::new_std(&ds);
    HASH_COUNTER.reset();
    let start = Instant::now();
    let witness_size = args.witness_size.unwrap_or(vector.len());
    let witness = config.commit(&mut state, &vector[..witness_size])?;
    let commitment = start.elapsed();
    let point = state.verifier_message_vec::<Field256>(args.variables);
    let linear = MultilinearExtension::new(point.clone());
    let start = Instant::now();
    let value = linear.evaluate(&Identity::new(), vector);
    let evaluation = start.elapsed();
    let form = TensorLinearForm::multilinear(&point);
    let start = Instant::now();
    config.prove(&mut state, witness, &form, value)?;
    let opening = start.elapsed();
    let proof = state.proof();
    finish(
        args,
        &proof,
        Timings {
            commitment,
            evaluation,
            opening,
        },
        prefix_oracles(config, args),
        |proof| {
            let mut state = VerifierState::new_std(&ds, proof);
            let commitment = config.receive_commitment(&mut state)?;
            let point = state.verifier_message_vec::<Field256>(args.variables);
            config.verify(
                &mut state,
                commitment,
                &TensorLinearForm::multilinear(&point),
                value,
            )?;
            state.check_eof()
        },
    )
}

fn plain(
    args: &Args,
    parameters: &ProtocolParameters,
    vector: &[Field256],
) -> VerificationResult<Measurement> {
    let config = WhirConfig::<Identity<Field256>>::new(vector.len(), parameters);
    verify!(config.check_max_pow_bits(0.0.into()));
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut state = ProverState::new_std(&ds);
    HASH_COUNTER.reset();
    let start = Instant::now();
    let witness = config.commit(&mut state, &[vector]);
    let commitment = start.elapsed();
    let point = state.verifier_message_vec::<Field256>(args.variables);
    let linear = MultilinearExtension::new(point);
    let start = Instant::now();
    let value = linear.evaluate(&Identity::new(), vector);
    let evaluation = start.elapsed();
    let start = Instant::now();
    let _ = config.prove(
        &mut state,
        vec![Cow::Borrowed(vector)],
        vec![Cow::Owned(witness)],
        vec![Box::new(linear)],
        Cow::Borrowed(&[value]),
    );
    let opening = start.elapsed();
    let proof = state.proof();
    finish(
        args,
        &proof,
        Timings {
            commitment,
            evaluation,
            opening,
        },
        baseline_oracles("plain", &config),
        |proof| {
            let mut state = VerifierState::new_std(&ds, proof);
            let commitment = config.receive_commitment(&mut state)?;
            let point = state.verifier_message_vec::<Field256>(args.variables);
            let linear = MultilinearExtension::new(point);
            config
                .verify(&mut state, &[&commitment], &[value])?
                .verify([&linear as &dyn LinearForm<Field256>])?;
            state.check_eof()
        },
    )
}

fn legacy(
    args: &Args,
    parameters: &ProtocolParameters,
    vector: &[Field256],
) -> VerificationResult<Measurement> {
    let config = LegacyConfig::<Field256>::new(args.variables, parameters);
    verify!(config.blinded_polynomial.check_max_pow_bits(0.0.into()));
    verify!(config.blinding_polynomial.check_max_pow_bits(0.0.into()));
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut state = ProverState::new_std(&ds);
    HASH_COUNTER.reset();
    let start = Instant::now();
    let witness = config.commit(&mut state, &[vector]);
    let commitment = start.elapsed();
    let point = state.verifier_message_vec::<Field256>(args.variables);
    let linear = MultilinearExtension::new(point);
    let start = Instant::now();
    let value = linear.evaluate(&Identity::new(), vector);
    let evaluation = start.elapsed();
    let start = Instant::now();
    config.prove(
        &mut state,
        vec![Cow::Borrowed(vector)],
        witness,
        vec![Box::new(linear)],
        Cow::Borrowed(&[value]),
    );
    let opening = start.elapsed();
    let proof = state.proof();
    finish(
        args,
        &proof,
        Timings {
            commitment,
            evaluation,
            opening,
        },
        {
            let mut oracles = baseline_oracles("legacy_witness", &config.blinded_polynomial);
            oracles.extend(baseline_oracles(
                "legacy_masks",
                &config.blinding_polynomial,
            ));
            oracles
        },
        |proof| {
            let mut state = VerifierState::new_std(&ds, proof);
            let commitments = config.receive_commitments(&mut state)?;
            let point = state.verifier_message_vec::<Field256>(args.variables);
            let linear = MultilinearExtension::new(point);
            let forms = [&linear as &dyn LinearForm<Field256>];
            config
                .verify(&mut state, &forms, &[value], &commitments)?
                .verify(forms)?;
            state.check_eof()
        },
    )
}

fn finish(
    args: &Args,
    proof: &Proof,
    timings: Timings,
    oracles: Vec<OracleParameters>,
    mut verify: impl FnMut(&Proof) -> VerificationResult<()>,
) -> VerificationResult<Measurement> {
    let prover_hashes = HASH_COUNTER.get();
    verify(proof)?;
    HASH_COUNTER.reset();
    let start = Instant::now();
    for _ in 0..args.verifier_repetitions {
        verify(proof)?;
    }
    let verification_ms =
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(args.verifier_repetitions);
    let verifier_hashes = HASH_COUNTER.get() as f64 / f64::from(args.verifier_repetitions);
    #[cfg(feature = "parallel")]
    let rayon_threads = current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let rayon_threads = 1;
    Ok(Measurement {
        mode: args.mode,
        variables: args.variables,
        witness_size: args.witness_size.unwrap_or(1usize << args.variables),
        run: 0,
        field: "BN254 scalar",
        field_modulus: Field256::MODULUS.to_string(),
        hash: "SHA-256",
        security_target_bits: args.security_bits,
        baseline_unique_decoding: (!matches!(args.mode, Mode::Prefix))
            .then_some(args.unique_decoding),
        candidate_security_certified: false,
        verifier_repetitions: args.verifier_repetitions,
        folds: args.folds,
        queries: matches!(args.mode, Mode::Prefix).then_some(args.queries),
        ood: matches!(args.mode, Mode::Prefix).then_some(args.ood),
        log_inv_rate: args.log_inv_rate,
        rayon_threads,
        oracles,
        oracle_metadata_scope:
            "Configured IRS draws; legacy initial root also has shared-index gamma openings.",
        commitment_ms: timings.commitment.as_secs_f64() * 1000.0,
        claim_evaluation_ms: timings.evaluation.as_secs_f64() * 1000.0,
        opening_ms: timings.opening.as_secs_f64() * 1000.0,
        verification_ms,
        proof_bytes: proof.narg_string.len() + proof.hints.len(),
        transcript_bytes: proof.narg_string.len(),
        hint_bytes: proof.hints.len(),
        prover_hashes,
        verifier_hashes,
    })
}
