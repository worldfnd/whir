#![allow(clippy::wildcard_imports)]

use ark_ff::{
    AdditiveGroup, Field, Fp3, Fp3Config, Fp64, MontBackend, MontConfig, MontFp, PrimeField,
};
use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
use divan::{black_box, counter::ItemsCount, Bencher};
use whir::algebra::fields::{
    goldilocks::{Acc, Goldilocks, GoldilocksReduce, MODULUS},
    Field64 as CustomF, Field64_3 as CustomF3,
};

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
#[small_subgroup_base = "15"]
#[small_subgroup_power = "1"]
struct ArkGoldilocksConfig;
type ArkF = Fp64<MontBackend<ArkGoldilocksConfig, 1>>;

struct ArkF3Config;
impl Fp3Config for ArkF3Config {
    type Fp = ArkF;

    const NONRESIDUE: Self::Fp = MontFp!("2");

    const FROBENIUS_COEFF_FP3_C1: &'static [Self::Fp] = &[
        MontFp!("1"),
        MontFp!("4294967295"),
        MontFp!("18446744065119617025"),
    ];

    const FROBENIUS_COEFF_FP3_C2: &'static [Self::Fp] = &[
        MontFp!("1"),
        MontFp!("18446744065119617025"),
        MontFp!("4294967295"),
    ];

    const TWO_ADICITY: u32 = 32;

    const QUADRATIC_NONRESIDUE_TO_T: Fp3<Self> =
        Fp3::new(MontFp!("5944137876247729999"), MontFp!("0"), MontFp!("0"));

    #[allow(clippy::unreadable_literal)]
    const TRACE_MINUS_ONE_DIV_TWO: &'static [u64] =
        &[0x80000002fffffffe, 0x80000002fffffffc, 0x7ffffffe];
}
type ArkF3 = Fp3<ArkF3Config>;

const N: usize = 1024;

fn random_raw(seed: u64, len: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen::<u64>() % MODULUS).collect()
}

fn to_field<F: PrimeField>(raw: &[u64]) -> Vec<F> {
    raw.iter().map(|&x| F::from(x)).collect()
}

fn to_fp3<F: PrimeField, C: Fp3Config<Fp = F>>(raw: &[u64]) -> Vec<Fp3<C>> {
    raw.chunks_exact(3)
        .map(|c| Fp3::new(F::from(c[0]), F::from(c[1]), F::from(c[2])))
        .collect()
}

#[divan::bench_group(name = "add")]
mod bench_add {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(1, N));
        let b = to_field::<CustomF>(&random_raw(2, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] + b[i];
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(1, N));
        let b = to_field::<ArkF>(&random_raw(2, N));
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] + b[i];
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "sub")]
mod bench_sub {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(1, N));
        let b = to_field::<CustomF>(&random_raw(2, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] - b[i];
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(1, N));
        let b = to_field::<ArkF>(&random_raw(2, N));
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] - b[i];
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "mul")]
mod bench_mul {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(1, N));
        let b = to_field::<CustomF>(&random_raw(2, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i];
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(1, N));
        let b = to_field::<ArkF>(&random_raw(2, N));
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i];
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "mul_chain")]
mod bench_mul_chain {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(3, N));
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut acc = CustomF::ONE;
            for &x in &a {
                acc *= x;
            }
            black_box(acc)
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(3, N));
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut acc = ArkF::ONE;
            for &x in &a {
                acc *= x;
            }
            black_box(acc)
        });
    }
}

#[divan::bench_group(name = "square")]
mod bench_square {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(4, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i].square();
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(4, N));
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i].square();
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "inverse")]
mod bench_inverse {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let raw: Vec<u64> = random_raw(5, N)
            .into_iter()
            .map(|x| if x == 0 { 1 } else { x })
            .collect();
        let a = to_field::<CustomF>(&raw);
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i].inverse().unwrap();
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let raw: Vec<u64> = random_raw(5, N)
            .into_iter()
            .map(|x| if x == 0 { 1 } else { x })
            .collect();
        let a = to_field::<ArkF>(&raw);
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i].inverse().unwrap();
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "mul_add_add")]
mod bench_mul_add_add {
    use super::*;

    #[divan::bench]
    fn custom_fused(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(1, N));
        let b = to_field::<CustomF>(&random_raw(2, N));
        let c = to_field::<CustomF>(&random_raw(3, N));
        let d = to_field::<CustomF>(&random_raw(4, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = (a[i].mul_wide(b[i]) + c[i] + d[i]).reduce_u64();
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn custom_naive(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(1, N));
        let b = to_field::<CustomF>(&random_raw(2, N));
        let c = to_field::<CustomF>(&random_raw(3, N));
        let d = to_field::<CustomF>(&random_raw(4, N));
        let mut out = vec![CustomF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i] + c[i] + d[i];
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(1, N));
        let b = to_field::<ArkF>(&random_raw(2, N));
        let c = to_field::<ArkF>(&random_raw(3, N));
        let d = to_field::<ArkF>(&random_raw(4, N));
        let mut out = vec![ArkF::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i] + c[i] + d[i];
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "sum_of_products_8")]
mod bench_sum_of_products {
    use super::*;

    const GROUPS: usize = N / 8;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(10, N));
        let b = to_field::<CustomF>(&random_raw(11, N));
        let mut out = vec![CustomF::ZERO; GROUPS];
        bencher.counter(ItemsCount::new(GROUPS)).bench_local(|| {
            for (g, (ca, cb)) in a.chunks_exact(8).zip(b.chunks_exact(8)).enumerate() {
                let ca: &[CustomF; 8] = ca.try_into().unwrap();
                let cb: &[CustomF; 8] = cb.try_into().unwrap();
                out[g] = CustomF::sum_of_products(ca, cb);
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(10, N));
        let b = to_field::<ArkF>(&random_raw(11, N));
        let mut out = vec![ArkF::ZERO; GROUPS];
        bencher.counter(ItemsCount::new(GROUPS)).bench_local(|| {
            for (g, (ca, cb)) in a.chunks_exact(8).zip(b.chunks_exact(8)).enumerate() {
                let ca: &[ArkF; 8] = ca.try_into().unwrap();
                let cb: &[ArkF; 8] = cb.try_into().unwrap();
                out[g] = ArkF::sum_of_products(ca, cb);
            }
            black_box(&out);
        });
    }
}

#[divan::bench_group(name = "dot_product")]
mod bench_dot_product {
    use super::*;

    #[divan::bench]
    fn custom_acc(bencher: Bencher) {
        let a = to_field::<CustomF>(&random_raw(20, N));
        let b = to_field::<CustomF>(&random_raw(21, N));
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut acc = Goldilocks::<Acc>::ZERO;
            for (&x, &y) in a.iter().zip(&b) {
                acc += x.mul_wide(y);
            }
            black_box(acc.reduce_u64())
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_field::<ArkF>(&random_raw(20, N));
        let b = to_field::<ArkF>(&random_raw(21, N));
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            let mut acc = ArkF::ZERO;
            for i in 0..N {
                acc += a[i] * b[i];
            }
            black_box(acc)
        });
    }
}

#[divan::bench_group(name = "fp3_mul")]
mod bench_fp3_mul {
    use super::*;

    #[divan::bench]
    fn custom(bencher: Bencher) {
        let a = to_fp3::<CustomF, _>(&random_raw(30, N * 3));
        let b = to_fp3::<CustomF, _>(&random_raw(31, N * 3));
        let mut out = vec![CustomF3::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i];
            }
            black_box(&out);
        });
    }

    #[divan::bench]
    fn arkworks(bencher: Bencher) {
        let a = to_fp3::<ArkF, ArkF3Config>(&random_raw(30, N * 3));
        let b = to_fp3::<ArkF, ArkF3Config>(&random_raw(31, N * 3));
        let mut out = vec![ArkF3::ZERO; N];
        bencher.counter(ItemsCount::new(N)).bench_local(|| {
            for i in 0..N {
                out[i] = a[i] * b[i];
            }
            black_box(&out);
        });
    }
}

fn main() {
    divan::main();
}
