use std::borrow::Cow;

use ark_std::rand::distributions::{Distribution, Standard};
use divan::{black_box, AllocProfiler, Bencher};
use spongefish::Codec;
use whir::algebra::embedding::{Basefield, Embedding};
use whir::algebra::fields::Field64_3;
use whir::algebra::linear_form::LinearForm;
use whir::buffer::{ActiveBuffer, BufferOps};
use whir::cmdline_utils::AvailableHash::Blake3;
use whir::parameters::ProtocolParameters;
use whir::protocols::whir::Config;
use whir::transcript::codecs::Empty;
use whir::transcript::{DomainSeparator, ProverState};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 16, 1 << 18, 1 << 20, 1 << 21];

type WhirEmbedding = Basefield<Field64_3>;

/// run whir as low degree test, no constraints(linear forms) or evaluations
#[divan::bench(args = SIZES)]
fn whir_ldt(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let vector = (0..size)
                .map(<WhirEmbedding as Embedding>::Source::from)
                .collect::<Vec<_>>();
            let input = ActiveBuffer::from_slice(&vector);
            let params = ProtocolParameters {
                security_level: 128,
                pow_bits: 20,
                initial_folding_factor: 4,
                folding_factor: 4,
                unique_decoding: false,
                starting_log_inv_rate: 1,
                batch_size: 1,
                hash_id: Blake3.hash_id(),
            };
            (input, params)
        })
        .bench_values(|(input, params)| {
            run_whir::<WhirEmbedding>(input, params, vec![], Cow::Borrowed(&vec![]));
        });
}

fn run_whir<'a, M: Embedding + Default>(
    input: ActiveBuffer<M::Source>,
    params: ProtocolParameters,
    linear_forms: Vec<Box<dyn LinearForm<M::Target>>>,
    evaluations: Cow<'a, [M::Target]>,
) where
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    M::Target: Codec,
{
    let config = Config::<M>::new(input.len(), &params);
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("Benchmark"))
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);
    let witness = config.commit(&mut prover_state, &[&input]);
    let result = config.prove(
        &mut prover_state,
        &[&input],
        vec![&witness],
        linear_forms,
        evaluations,
    );
    let _ = black_box(result);
}

fn main() {
    divan::main();
}
