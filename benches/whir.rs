use std::borrow::Cow;

use ark_std::rand::distributions::{Distribution, Standard};
use divan::{black_box, AllocProfiler, Bencher};
use spongefish::Codec;
use whir::{
    algebra::{
        embedding::{Basefield, Embedding},
        fields::Field64_3,
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
    },
    buffer::{ActiveBuffer, BufferOps},
    cmdline_utils::AvailableHash::Blake3,
    parameters::ProtocolParameters,
    protocols::whir::Config,
    transcript::{codecs::Empty, DomainSeparator, ProverState},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 16, 1 << 18, 1 << 20, 1 << 21];
const PROTOCOL_PARAMS: ProtocolParameters = ProtocolParameters {
    security_level: 128,
    pow_bits: 20,
    initial_folding_factor: 4,
    folding_factor: 4,
    unique_decoding: false,
    starting_log_inv_rate: 1,
    batch_size: 1,
    hash_id: Blake3.hash_id(),
};

type WhirEmbedding = Basefield<Field64_3>;
type Target = <WhirEmbedding as Embedding>::Target;

/// run whir as low degree test, no constraints(linear forms) or evaluations
#[divan::bench(args = SIZES)]
fn whir_ldt(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let vector = (0..size)
                .map(<WhirEmbedding as Embedding>::Source::from)
                .collect::<Vec<_>>();
            ActiveBuffer::from_slice(&vector)
        })
        .bench_values(|input| {
            run_whir::<WhirEmbedding>(&input, vec![], Cow::Borrowed(&[]));
        });
}

/// run whir as polynomial commitment scheme
#[divan::bench(args = SIZES)]
fn whir_pcs(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let num_variables = size.trailing_zeros() as usize;
            let vector = (0..size)
                .map(<WhirEmbedding as Embedding>::Source::from)
                .collect::<Vec<_>>();
            let input = ActiveBuffer::from_slice(&vector);
            let points: Vec<_> = (0..2u64)
                .map(|i| vec![Target::from(i); num_variables])
                .collect();
            let mut evaluations = Vec::new();
            for point in &points {
                let linear_form = MultilinearExtension::new(point.clone());
                evaluations.push(linear_form.evaluate(&WhirEmbedding::default(), &vector));
            }
            let linear_forms: Vec<Box<dyn LinearForm<Target>>> = points
                .iter()
                .map(|p| {
                    Box::new(MultilinearExtension::new(p.clone())) as Box<dyn LinearForm<Target>>
                })
                .collect();

            (input, linear_forms, evaluations)
        })
        .bench_values(|(input, linear_forms, evaluations)| {
            run_whir::<WhirEmbedding>(&input, linear_forms, Cow::Borrowed(&evaluations));
        });
}

fn run_whir<M: Embedding + Default>(
    input: &ActiveBuffer<M::Source>,
    linear_forms: Vec<Box<dyn LinearForm<M::Target>>>,
    evaluations: Cow<'_, [M::Target]>,
) where
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    M::Target: Codec,
{
    let config = Config::<M>::new(input.len(), &PROTOCOL_PARAMS);
    let ds = DomainSeparator::protocol(&config)
        .session(&"Benchmark".to_string())
        .instance(&Empty);
    let mut prover_state = ProverState::new_std(&ds);
    let witness = config.commit(&mut prover_state, &[input]);
    let result = config.prove(
        &mut prover_state,
        &[input],
        vec![&witness],
        linear_forms,
        evaluations,
    );
    let _ = black_box(result);
}

fn main() {
    divan::main();
}
