use std::{str::FromStr, time::Instant};

use ark_std::rand::distributions::{Distribution, Standard};
use clap::Parser;
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field128, Field192, Field256, Field64, Field64_2, Field64_3},
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
    },
    buffer::Buffer,
    engines::EngineId,
    hash::{self, HASH_COUNTER},
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
    DecodingRegime, KneeWeight, Mode, Params, ProtocolConfig, RateSchedule,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "128")]
    security_level: u32,

    /// Maximum proof of work difficulty in bits.
    #[arg(short = 'p', long, default_value = "20")]
    pow_bits: u32,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(long = "linear-constraints", default_value = "0")]
    num_linear_constraints: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: u32,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    /// Reed–Solomon decoding regime: Unique, Johnson, or Capacity.
    #[arg(long = "decoding-regime", default_value = "Johnson", value_parser = parse_regime)]
    decoding_regime: DecodingRegime,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,

    /// Rate schedule: `stepping`, `capped:<max log inv rate>`, or `adaptive:<knee weight>`.
    #[arg(long = "schedule", default_value = "adaptive:0.5", value_parser = parse_schedule)]
    schedule: RateSchedule,

    /// Run in zero-knowledge mode. Requires Johnson decoding.
    #[arg(long = "zk")]
    zk: bool,
}

#[derive(Debug, Clone, Copy)]
enum AvailableFields {
    Goldilocks1, // Just Goldilocks
    Goldilocks2, // Quadratic extension of Goldilocks
    Goldilocks3, // Cubic extension of Goldilocks
    Field128,    // 128-bit prime field
    Field192,    // 192-bit prime field
    Field256,    // 256-bit prime field
}

impl FromStr for AvailableFields {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Field128" => Ok(Self::Field128),
            "Field192" => Ok(Self::Field192),
            "Field256" => Ok(Self::Field256),
            "Goldilocks1" => Ok(Self::Goldilocks1),
            "Goldilocks2" => Ok(Self::Goldilocks2),
            "Goldilocks3" => Ok(Self::Goldilocks3),
            _ => Err(format!("Invalid field: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AvailableHash {
    Sha2,
    Sha3,
    Keccak,
    Blake3,
}

impl AvailableHash {
    const fn hash_id(self) -> EngineId {
        match self {
            Self::Sha2 => hash::SHA2,
            Self::Sha3 => hash::SHA3,
            Self::Keccak => hash::KECCAK,
            Self::Blake3 => hash::BLAKE3,
        }
    }
}

impl FromStr for AvailableHash {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Sha2" => Ok(Self::Sha2),
            "Sha3" => Ok(Self::Sha3),
            "Keccak" => Ok(Self::Keccak),
            "Blake3" => Ok(Self::Blake3),
            _ => Err(format!(
                "Invalid hash: {s}, options are: Sha2, Sha3, Keccak, Blake3"
            )),
        }
    }
}

fn parse_regime(s: &str) -> Result<DecodingRegime, String> {
    match s {
        "Unique" => Ok(DecodingRegime::Unique),
        "Johnson" => Ok(DecodingRegime::Johnson),
        "Capacity" => Ok(DecodingRegime::Capacity),
        _ => Err(format!(
            "invalid decoding regime: {s}, options are: Unique, Johnson, Capacity"
        )),
    }
}

fn parse_schedule(s: &str) -> Result<RateSchedule, String> {
    match s.split_once(':') {
        None if s == "stepping" => Ok(RateSchedule::Stepping),
        Some(("capped", cap)) => cap
            .parse()
            .map(RateSchedule::Capped)
            .map_err(|e| format!("invalid cap: {e}")),
        Some(("adaptive", weight)) => weight
            .parse::<f64>()
            .map(|w| RateSchedule::Adaptive(KneeWeight::from(w)))
            .map_err(|e| format!("invalid knee weight: {e}")),
        _ => Err(format!(
            "invalid schedule: {s}, options are: stepping, capped:<u32>, adaptive:<f64>"
        )),
    }
}

fn main() {
    use AvailableFields as AF;
    let args = Args::parse();
    let field = args.field;

    // Dispatch on embedding
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
    let num_variables = args.num_variables;
    let reps = args.verifier_repetitions;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;

    assert!(
        num_evaluations + num_linear_constraints > 0,
        "WHIR requires at least one evaluation or linear constraint"
    );

    let num_coeffs = 1 << num_variables;

    let params = ProtocolConfig::<M>::derive(Params {
        mode: if args.zk {
            Mode::ZeroKnowledge
        } else {
            Mode::Standard
        },
        regime: args.decoding_regime,
        bits: args.security_level,
        pow: args.pow_bits,
        hash: args.hash.hash_id(),
        size: num_coeffs,
        rate: args.rate,
        folds: [args.first_round_folding_factor, args.folding_factor],
        schedule: args.schedule,
    })
    .unwrap_or_else(|e| panic!("invalid parameters: {e:?}"));

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS{}) 🌪️", if args.zk { " + ZK" } else { "" });
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{:?}", params.params());

    let embedding = M::default();
    let vector = (0..num_coeffs as u64)
        .map(M::Source::from)
        .collect::<Vec<_>>();

    let vector_buffer = Buffer::from(vector.as_slice());
    let whir_commit_time = Instant::now();
    let committed = params.commit(&mut prover_state, vector_buffer);
    let whir_commit_time = whir_commit_time.elapsed();

    // Allocate constraints
    let mut linear_forms: Vec<Box<dyn Evaluate<M>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Linear constraint
    for _ in 0..num_linear_constraints {
        let linear_form = Box::new(Covector {
            vector: (0..num_coeffs as u64).map(M::Target::from).collect(),
        });
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form);
    }

    // Evaluation constraint
    for x in 0..num_evaluations {
        let linear_form = Box::new(MultilinearExtension::new(vec![
            M::Target::from(x as u64);
            num_variables
        ]));
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form);
    }
    let linear_forms: Vec<&dyn LinearForm<M::Target>> = linear_forms
        .iter()
        .map(|w| w.as_ref() as &dyn LinearForm<M::Target>)
        .collect();

    let whir_prove_time = Instant::now();
    let _ = params.prove(&mut prover_state, committed, &linear_forms, &evaluations);
    let whir_prove_time = whir_prove_time.elapsed();

    let proof = prover_state.proof();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();
        let final_claim = params
            .verify(&mut verifier_state, commitment, &linear_forms, &evaluations)
            .unwrap();
        final_claim.verify(&linear_forms).unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}
