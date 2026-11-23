use std::{
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

use ark_std::rand::{distributions::Standard, prelude::Distribution};
use clap::Parser;
use serde::Serialize;
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field128, Field192, Field256, Field64, Field64_2, Field64_3},
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
    },
    buffer::Buffer,
    cmdline_utils::{AvailableFields, AvailableHash},
    hash::HASH_COUNTER,
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
    DecodingRegime, Mode, Params, ProtocolConfig, RateSchedule,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "128")]
    security_level: u32,

    #[arg(short = 'p', long, default_value = "20")]
    pow_bits: u32,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: u32,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "decoding-regime", default_value = "Johnson")]
    decoding_regime: DecodingRegime,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    security_level: u32,
    pow_bits: u32,
    starting_rate: u32,
    num_variables: usize,
    repetitions: usize,
    initial_folding_factor: usize,
    folding_factor: usize,
    decoding_regime: DecodingRegime,
    field: AvailableFields,
    hash: AvailableHash,

    // Whir
    whir_evaluations: usize,
    whir_argument_size: usize,
    whir_prover_time: Duration,
    whir_prover_hashes: usize,
    whir_verifier_time: Duration,
    whir_verifier_hashes: usize,
}

fn main() {
    use AvailableFields as AF;
    let args = Args::parse();
    let field = args.field;

    // Type reflection on field
    match field {
        AF::Goldilocks1 => run_whir::<Identity<Field64>>(&args),
        AF::Goldilocks2 => run_whir::<Basefield<Field64_2>>(&args),
        AF::Goldilocks3 => run_whir::<Basefield<Field64_3>>(&args),
        AF::Field128 => run_whir::<Identity<Field128>>(&args),
        AF::Field192 => run_whir::<Identity<Field192>>(&args),
        AF::Field256 => run_whir::<Identity<Field256>>(&args),
    }
}

#[allow(clippy::too_many_lines)]
fn run_whir<M>(args: &Args)
where
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    M: Embedding + Default,
    M::Target: Codec,
{
    let security_level = args.security_level;
    let pow_bits = args.pow_bits;
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let first_round_folding_factor = args.first_round_folding_factor;
    let decoding_regime = args.decoding_regime;

    std::fs::create_dir_all("outputs").unwrap();

    let num_coeffs = 1 << num_variables;

    let vector = (0..num_coeffs as u64)
        .map(M::Source::from)
        .collect::<Vec<_>>();

    let (
        whir_prover_time,
        whir_argument_size,
        whir_prover_hashes,
        whir_verifier_time,
        whir_verifier_hashes,
    ) = {
        let params = ProtocolConfig::<M>::derive(Params {
            mode: Mode::Standard,
            regime: decoding_regime,
            bits: security_level,
            pow: pow_bits,
            hash: args.hash.hash_id(),
            size: num_coeffs,
            rate: starting_rate,
            folds: [first_round_folding_factor, folding_factor],
            schedule: RateSchedule::Stepping,
        })
        .unwrap_or_else(|e| panic!("invalid parameters: {e:?}"));

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Example at {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);

        let points: Vec<_> = (0..args.num_evaluations)
            .map(|i| vec![M::Target::from(i as u64); num_variables])
            .collect();

        let embedding = M::default();
        let mut weights: Vec<Box<dyn Evaluate<M>>> = Vec::new();
        let mut evaluations = Vec::new();

        for point in &points {
            let linear_form = MultilinearExtension::new(point.clone());
            evaluations.push(linear_form.evaluate(&embedding, &vector));
            weights.push(Box::new(linear_form));
        }
        let weights: Vec<&dyn LinearForm<M::Target>> = weights
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<M::Target>)
            .collect();

        HASH_COUNTER.reset();
        let whir_prover_time = Instant::now();

        let committed = params.commit(&mut prover_state, Buffer::from(vector.as_slice()));
        let _ = params.prove(&mut prover_state, committed, &weights, &evaluations);

        let whir_prover_time = whir_prover_time.elapsed();
        let proof = prover_state.proof();
        let whir_argument_size = proof.narg_string.len() + proof.hints.len();
        let whir_prover_hashes = HASH_COUNTER.get();

        HASH_COUNTER.reset();
        let whir_verifier_time = Instant::now();
        for _ in 0..reps {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);

            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            let final_claim = params
                .verify(&mut verifier_state, commitment, &weights, &evaluations)
                .unwrap();
            final_claim.verify(&weights).unwrap();
        }

        let whir_verifier_time = whir_verifier_time.elapsed();
        let whir_verifier_hashes = HASH_COUNTER.get() / reps;

        (
            whir_prover_time,
            whir_argument_size,
            whir_prover_hashes,
            whir_verifier_time,
            whir_verifier_hashes,
        )
    };

    let output = BenchmarkOutput {
        security_level,
        pow_bits,
        starting_rate,
        num_variables,
        repetitions: reps,
        initial_folding_factor: first_round_folding_factor,
        folding_factor,
        decoding_regime,
        field: args.field,
        hash: args.hash,

        // Whir
        whir_evaluations: args.num_evaluations,
        whir_prover_time,
        whir_argument_size,
        whir_prover_hashes,
        whir_verifier_time,
        whir_verifier_hashes,
    };

    let mut out_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("outputs/bench_output.json")
        .unwrap();
    writeln!(out_file, "{}", serde_json::to_string(&output).unwrap()).unwrap();
}
