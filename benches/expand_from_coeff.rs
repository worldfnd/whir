use divan::{black_box, AllocProfiler, Bencher};
use whir::algebra::{fields::Field64, ntt::NttEngine, random_vector};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

//
// Test cases with tuple entries:
//      - polynomial sizes defined as exponents of 2,
//      - RS code expansion factors, and
//      - interleaved bloc size exponent of 2
//
const TEST_CASES: &[(usize, usize, usize)] = &[
    (16, 2, 2),
    (18, 2, 2),
    (20, 2, 3),
    (16, 4, 3),
    (18, 4, 3),
    (20, 4, 4),
    (22, 4, 4),
];

#[divan::bench(args = TEST_CASES)]
fn interleaved_rs_encode(bencher: Bencher, case: &(usize, usize, usize)) {
    bencher
        .with_inputs(|| {
            let (exp, expansion, coset_sz) = *case;
            let message_length = 1 << (exp - coset_sz);
            let num_messages = 1 << coset_sz;
            let mut rng = ark_std::rand::thread_rng();
            let coeffs: Vec<Vec<Field64>> = (0..num_messages)
                .map(|_| random_vector(&mut rng, message_length))
                .collect();
            let engine = NttEngine::<Field64>::new_from_fftfield();
            (engine, coeffs, expansion)
        })
        .bench_values(|(engine, coeffs, expansion)| {
            let coeffs_refs = coeffs.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
            black_box(engine.interleaved_encode_slices(
                &coeffs_refs,
                &[],
                coeffs[0].len() * expansion,
            ))
        });
}

fn main() {
    divan::main();
}
