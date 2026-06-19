use std::{
    borrow::Cow,
    fs,
    path::{Path, PathBuf},
};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use whir::{
    algebra::{embedding::Identity, fields::Field256, linear_form::LinearForm},
    hash,
    parameters::ProtocolParameters,
    protocols::whir::Config as WhirConfig,
    transcript::{codecs::Empty, DomainSeparator, Proof, ProverState, VerifierState},
};

use whir::buffer::{ActiveBuffer, BufferOps};

type F = Field256;
type M = Identity<F>;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Produce a WHIR proof and verify it from a serialized artifact"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Prove(RoundtripArgs),
    Verify {
        #[arg(long)]
        input: PathBuf,
    },
}

#[derive(Parser, Debug)]
struct RoundtripArgs {
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 16)]
    log_size: usize,

    #[arg(long, default_value_t = 4)]
    fold: usize,

    #[arg(long, default_value_t = 1)]
    rate: usize,

    #[arg(long, default_value_t = 20)]
    pow_bits: usize,

    #[arg(long, default_value_t = 128)]
    security_level: usize,
}

#[derive(Serialize, Deserialize)]
struct Artifact {
    log_size: usize,
    fold: usize,
    rate: usize,
    pow_bits: usize,
    security_level: usize,
    proof: Proof,
    proof_bytes: usize,
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Prove(args) => prove(&args),
        Command::Verify { input } => verify(&input),
    }
}

fn prove(args: &RoundtripArgs) {
    assert!(args.fold <= args.log_size, "fold must be <= log_size");
    let size = 1usize << args.log_size;
    let whir_params = protocol_parameters(args.security_level, args.pow_bits, args.fold, args.rate);
    let params = WhirConfig::<M>::new(size, &whir_params);
    let ds = DomainSeparator::protocol(&params)
        .session(&"gpu proof cpu verify")
        .instance(&Empty);

    let vector = input_vector(size);
    let vector_buffer = ActiveBuffer::from_slice(&vector);

    let mut prover_state = ProverState::new_std(&ds);
    let witness = params.commit(&mut prover_state, &[&vector_buffer]);
    let _ = params.prove(
        &mut prover_state,
        &[&vector_buffer],
        vec![&witness],
        vec![],
        Cow::Owned(vec![]),
    );
    let proof = prover_state.proof();
    let proof_bytes = proof.narg_string.len() + proof.hints.len();
    let artifact = Artifact {
        log_size: args.log_size,
        fold: args.fold,
        rate: args.rate,
        pow_bits: args.pow_bits,
        security_level: args.security_level,
        proof,
        proof_bytes,
    };
    let encoded = serde_json::to_vec_pretty(&artifact).expect("serialize proof artifact");
    fs::write(&args.output, encoded).expect("write proof artifact");
    println!(
        "wrote proof artifact backend={} log_size={} fold={} rate={} pow_bits={} proof_bytes={}",
        backend_name(),
        args.log_size,
        args.fold,
        args.rate,
        args.pow_bits,
        proof_bytes
    );
}

fn verify(input: &Path) {
    let bytes = fs::read(input).expect("read proof artifact");
    let artifact: Artifact = serde_json::from_slice(&bytes).expect("decode proof artifact");
    let size = 1usize << artifact.log_size;
    let whir_params = protocol_parameters(
        artifact.security_level,
        artifact.pow_bits,
        artifact.fold,
        artifact.rate,
    );
    let params = WhirConfig::<M>::new(size, &whir_params);
    let ds = DomainSeparator::protocol(&params)
        .session(&"gpu proof cpu verify")
        .instance(&Empty);

    let mut verifier_state = VerifierState::new_std(&ds, &artifact.proof);
    let commitment = params
        .receive_commitment(&mut verifier_state)
        .expect("receive commitment");
    let final_claim = params
        .verify(&mut verifier_state, &[&commitment], &[])
        .expect("verify WHIR proof");
    let no_forms: Vec<&dyn LinearForm<F>> = Vec::new();
    final_claim.verify(no_forms).expect("verify final claim");
    verifier_state.check_eof().expect("proof EOF");
    println!(
        "verified proof artifact backend={} log_size={} fold={} rate={} pow_bits={} proof_bytes={}",
        backend_name(),
        artifact.log_size,
        artifact.fold,
        artifact.rate,
        artifact.pow_bits,
        artifact.proof_bytes
    );
}

const fn protocol_parameters(
    security_level: usize,
    pow_bits: usize,
    fold: usize,
    rate: usize,
) -> ProtocolParameters {
    ProtocolParameters {
        security_level,
        pow_bits,
        initial_folding_factor: fold,
        folding_factor: fold,
        unique_decoding: false,
        starting_log_inv_rate: rate,
        batch_size: 1,
        hash_id: hash::SHA2,
    }
}

fn input_vector(size: usize) -> Vec<F> {
    (0..size).map(|i| F::from(i as u64)).collect()
}

#[cfg(all(feature = "metal", target_os = "macos"))]
const fn backend_name() -> &'static str {
    "gpu-metal"
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
const fn backend_name() -> &'static str {
    "cpu"
}
