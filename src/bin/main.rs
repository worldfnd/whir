use std::{borrow::Cow, time::Instant};

#[cfg(feature = "rs_in_order")]
use ark_ff::Field;
use ark_std::rand::distributions::{Distribution, Standard};
use clap::Parser;
use whir::{
    algebra::{
        buffer::ActiveBuffer,
        embedding::{Basefield, Embedding, Identity},
        fields::{Field128, Field192, Field256, Field64, Field64_2, Field64_3},
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
    },
    bits::Bits,
    cmdline_utils::{AvailableFields, AvailableHash},
    hash::HASH_COUNTER,
    parameters::ProtocolParameters,
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "128")]
    security_level: usize,

    /// Maximum proof of work difficulty in bits.
    #[arg(short = 'p', long, default_value = "20")]
    pow_bits: usize,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(long = "linear-constraints", default_value = "0")]
    num_linear_constraints: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    /// Restrict PCS to the Unique Decoding regime. LDT is always UD.
    #[arg(long = "unique-decoding", default_value_t = false)]
    unique_decoding: bool,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,

    #[arg(long = "zk")]
    zk: bool,
}

fn main() {
    use AvailableFields as AF;
    let args = Args::parse();
    let field = args.field;

    // Dispatch on embedding
    if args.zk {
        #[cfg(not(feature = "rs_in_order"))]
        panic!("ZK requires --features rs_in_order");
        #[cfg(feature = "rs_in_order")]
        match field {
            AF::Goldilocks1 => run_whir_zk::<Field64>(&args),
            AF::Goldilocks2 => run_whir_zk::<Field64_2>(&args),
            AF::Goldilocks3 => run_whir_zk::<Field64_3>(&args),
            AF::Field128 => run_whir_zk::<Field128>(&args),
            AF::Field192 => run_whir_zk::<Field192>(&args),
            AF::Field256 => run_whir_zk::<Field256>(&args),
        }
    } else {
        match field {
            AF::Goldilocks1 => run_whir::<Identity<Field64>>(&args),
            AF::Goldilocks2 => run_whir::<Basefield<Field64_2>>(&args),
            AF::Goldilocks3 => run_whir::<Basefield<Field64_3>>(&args),
            AF::Field128 => run_whir::<Identity<Field128>>(&args),
            AF::Field192 => run_whir::<Identity<Field192>>(&args),
            AF::Field256 => run_whir::<Identity<Field256>>(&args),
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run_whir<M>(args: &Args)
where
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    M: Embedding + Default,
    M::Target: Codec,
{
    use whir::protocols::whir::Config;

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits;
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let unique_decoding = args.unique_decoding;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;
    let hash_id = args.hash.hash_id();

    if num_evaluations + num_linear_constraints == 0 {
        println!("No constraints specified, running as low-degree-test.");
    }

    let num_coeffs = 1 << num_variables;

    let whir_params = ProtocolParameters {
        security_level,
        pow_bits,
        initial_folding_factor: first_round_folding_factor,
        folding_factor,
        unique_decoding,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = Config::<M>::new(1 << num_variables, &whir_params);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
        println!("WARN: more PoW bits required than specified.");
    }

    let vector = (0..num_coeffs).map(M::Source::from).collect::<Vec<_>>();
    let vector_buffer = ActiveBuffer::from_slice(&vector);

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &[&vector_buffer]);
    let whir_commit_time = whir_commit_time.elapsed();

    // Allocate constraints
    let mut linear_forms: Vec<Box<dyn Evaluate<M>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<M::Target>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Linear constraint
    // We do these first to benefit from buffer recycling.
    for _ in 0..num_linear_constraints {
        let linear_form = Box::new(Covector {
            vector: (0..num_coeffs).map(M::Target::from).collect(),
        });
        evaluations.push(linear_form.evaluate(params.embedding(), &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| vec![M::Target::from(x as u64); num_variables])
        .collect();
    for point in &points {
        let linear_form = Box::new(MultilinearExtension::new(point.clone()));
        evaluations.push(linear_form.evaluate(params.embedding(), &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    let whir_prove_time = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        &[&vector_buffer],
        vec![&witness],
        prove_linear_forms,
        Cow::Borrowed(evaluations.as_slice()),
    );
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
            .verify(&mut verifier_state, &[&commitment], &evaluations)
            .unwrap();
        final_claim
            .verify(
                linear_forms
                    .iter()
                    .map(|w| w.as_ref() as &dyn LinearForm<M::Target>),
            )
            .unwrap();
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

#[cfg(feature = "rs_in_order")]
#[allow(clippy::too_many_lines)]
fn run_whir_zk<F>(args: &Args)
where
    Standard: Distribution<F>,
    F: Field + Codec,
{
    use whir::protocols::whir_zk::Config;

    let security_level = args.security_level;
    let pow_bits = args.pow_bits;
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;
    let hash_id = args.hash.hash_id();

    if num_evaluations + num_linear_constraints == 0 {
        println!("No constraints specified, running as low-degree-test.");
    }

    let num_coeffs = 1 << num_variables;

    let whir_params = ProtocolParameters {
        unique_decoding: args.unique_decoding,
        security_level,
        pow_bits,
        initial_folding_factor: first_round_folding_factor,
        folding_factor,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = Config::<F>::new(1 << num_variables, &whir_params, 1);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS + ZK) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params
        .blinded_commitment
        .check_max_pow_bits(Bits::new(whir_params.pow_bits as f64))
    {
        println!("WARN: more PoW bits required than specified.");
    }

    let embedding = Identity::<F>::new();
    let vector = (0..num_coeffs).map(F::from).collect::<Vec<_>>();

    // Allocate constraints
    let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<F>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Linear constraint
    // We do these first to benefit from buffer recycling.
    for _ in 0..num_linear_constraints {
        let linear_form = Box::new(Covector {
            vector: (0..num_coeffs).map(F::from).collect(),
        });
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| vec![F::from(x as u64); num_variables])
        .collect();
    for point in &points {
        let linear_form = Box::new(MultilinearExtension::new(point.clone()));
        evaluations.push(linear_form.evaluate(&embedding, &vector));
        linear_forms.push(linear_form.clone());
        prove_linear_forms.push(linear_form);
    }

    let whir_commit_time = Instant::now();
    let vector_buffer = ActiveBuffer::from_slice(&vector);
    let witness = params.commit(&mut prover_state, &[&vector_buffer]);
    let whir_commit_time = whir_commit_time.elapsed();

    let whir_prove_time = Instant::now();
    let _ = params.prove(
        &mut prover_state,
        &[&vector_buffer],
        vec![&witness],
        prove_linear_forms,
        Cow::Borrowed(&evaluations),
    );
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

    let weight_dyn_refs = linear_forms
        .iter()
        .map(|w| w.as_ref() as &dyn LinearForm<F>)
        .collect::<Vec<_>>();

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params.receive_commitments(&mut verifier_state, 1).unwrap();
        params
            .verify(
                &mut verifier_state,
                &weight_dyn_refs,
                &evaluations,
                &commitment,
            )
            .unwrap();
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
