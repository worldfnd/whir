#[cfg(all(feature = "metal", target_os = "macos"))]
mod bench {
    use divan::{black_box, AllocProfiler, Bencher};
    use whir::{
        algebra::{
            buffer::{BufferRead, CpuBuffer, MetalBuffer},
            fields::Field256 as F,
        },
        hash::{Hash, HashEngine, MetalSha2, Sha2},
    };

    #[global_allocator]
    static ALLOC: AllocProfiler = AllocProfiler::system();

    const FIELD_SIZES: &[usize] = &[1 << 8, 1 << 10];
    const HASH_ROWS: &[usize] = &[1 << 10, 1 << 12];
    const HASH_ROW_SIZE: usize = 128;

    #[divan::bench(args = FIELD_SIZES)]
    fn cpu_bn254_dot(bencher: Bencher, size: usize) {
        bencher
            .with_inputs(|| {
                let a = CpuBuffer::from_vec((0..size).map(|i| F::from(i as u64)).collect());
                let b = CpuBuffer::from_vec((0..size).map(|i| F::from((i + 7) as u64)).collect());
                (a, b)
            })
            .bench_values(|(a, b)| black_box(a.dot(&b)));
    }

    #[divan::bench(args = FIELD_SIZES)]
    fn metal_bn254_dot(bencher: Bencher, size: usize) {
        bencher
            .with_inputs(|| {
                let a = MetalBuffer::from_vec((0..size).map(|i| F::from(i as u64)).collect());
                let b = MetalBuffer::from_vec((0..size).map(|i| F::from((i + 7) as u64)).collect());
                (a, b)
            })
            .bench_values(|(a, b)| black_box(a.dot(&b)));
    }

    #[divan::bench(args = HASH_ROWS)]
    fn cpu_sha256_hash_many(bencher: Bencher, rows: usize) {
        bencher
            .with_inputs(|| {
                let input = (0..rows * HASH_ROW_SIZE)
                    .map(|i| (i & 0xff) as u8)
                    .collect::<Vec<_>>();
                let output = vec![Hash::default(); rows];
                (input, output)
            })
            .bench_values(|(input, mut output)| {
                Sha2::new().hash_many(HASH_ROW_SIZE, &input, &mut output);
                black_box(output)
            });
    }

    #[divan::bench(args = HASH_ROWS)]
    fn metal_sha256_hash_many(bencher: Bencher, rows: usize) {
        bencher
            .with_inputs(|| {
                let input = (0..rows * HASH_ROW_SIZE)
                    .map(|i| (i & 0xff) as u8)
                    .collect::<Vec<_>>();
                let output = vec![Hash::default(); rows];
                (input, output)
            })
            .bench_values(|(input, mut output)| {
                MetalSha2::new().hash_many(HASH_ROW_SIZE, &input, &mut output);
                black_box(output)
            });
    }

    pub fn main() {
        divan::main();
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    bench::main();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("metal_buffer benchmark requires macOS and --features metal");
}
