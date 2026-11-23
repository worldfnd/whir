use divan::{black_box, AllocProfiler, Bencher};
use whir::algebra::{
    fields::Field64 as F,
    sumcheck_kernels::{compute_sumcheck_polynomial, fold},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 16, 1 << 18, 1 << 20];

#[divan::bench(args = SIZES)]
fn sumcheck_first_round(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            // Reset everything on each iteration
            let a = (0..size).map(F::from).collect::<Vec<_>>();
            let b = (0..size).map(F::from).collect::<Vec<_>>();
            let folding_randomness = F::from(42);
            (a, b, folding_randomness)
        })
        .bench_values(|(mut a, mut b, folding_randomness)| {
            let poly = compute_sumcheck_polynomial(&a, &b);
            fold(&mut a, folding_randomness);
            fold(&mut b, folding_randomness);
            black_box((poly, a, b))
        });
}

fn main() {
    divan::main();
}
