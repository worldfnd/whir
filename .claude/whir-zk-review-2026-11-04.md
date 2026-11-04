# WHIR v1 private-mask experiment

Feature-gated construction for one fresh commitment and one adaptively chosen linear opening. Prefix masks hide oracle replies; committed quadratic masks hide sumcheck messages; code-switch carries mask constraints privately. Initial and subsequent main commitments include OOD checks. [Protocol](whir-zk-construction-2026-11-04.md) · [Security argument](whir-zk-security-2026-11-04.md).

Existing v1 behavior is unchanged. The experiment reuses RS encoding, matrix/Merkle commitments, query sampling, folding, and ordinary sumcheck verification. New code handles private packing, oracle rows, and their linear forms; no dependencies were added.

## Security scope

- Earlier protocol reviews cover the salted layout. Full-view privacy of the current unsalted variant remains unproved; these are model-assisted reviews, not an external audit.
- Existing security arguments require rechecking against the unsalted layout. Fixed-hash instantiation remains heuristic.
- BN254, $N=2^{20}$, five folds, rate $1/4$, base payload 2048: list uses 148 queries; unique uses 195. Conditional ideal-binding-oracle calculations give global interactive error below $2^{-129}$; noninteractive error also depends on oracle-query budgets.
- Commitment reuse, arbitrary parameter certification, and knowledge extraction are outside this experiment.

## Unsalted measurements

WHIR-only benchmarks using the v1 `complete_age_check` segment lengths: 657,980 (657,900 witness entries plus 80 outer-sumcheck mask coefficients) and 655,177. Payload values are synthetic; each segment gets one multilinear opening. Cells sum the two per-segment medians, not end-to-end circuit timings.

Darwin arm64, rustc 1.99.1, release fat LTO, SHA-256, five folds, rate $1/4$, no proof of work. Three runs per case; 20 verifier repetitions per run. Cells show **prove ms / verify ms**. Proof sizes sum the eight-thread medians.

| Construction | 1 thread | 4 threads | 8 threads | Proof KiB |
|---|---:|---:|---:|---:|
| Plain, list | 3746 / 1.68 | 1236 / 5.69 | 802 / 1.92 | 596 |
| Legacy hiding, list | 5289 / 107.72 | 1918 / 43.19 | 1324 / 37.52 | 1526 |
| Private masks, list | 2962 / 30.58 | 905 / 28.47 | 796 / 29.91 | 1497 |
| Plain, unique | 3785 / 3.69 | 1193 / 3.61 | 763 / 3.72 | 1466 |
| Private masks, unique | 2833 / 32.15 | 892 / 29.99 | 677 / 30.36 | 1888 |

Maximum per-segment process RSS at eight threads: private-mask list 545 MiB, legacy hiding 693 MiB, plain list 732 MiB. Legacy unique decoding is unsupported by v1.

CPU times varied between runs. A separate comparison alternated the saved salted and current unsalted binaries, reversed their order each iteration, discarded one warmup per case, and kept three measurements. Only the leaf layout and its domain-separator version differ. Both use 148 queries.

| Threads | Salted prove / verify ms | Unsalted prove / verify ms | Salted / unsalted proof KiB |
|---|---:|---:|---:|
| 1 | 4813 / 59.14 | 4518 / 52.15 | 1638 / 1492 |
| 4 | 1243 / 29.86 | 1223 / 29.64 | 1640 / 1493 |
| 8 | 683 / 29.37 | 601 / 28.12 | 1641 / 1491 |

The paired eight-thread run shows 12% less proving time, 4% less verification time, and 9% less proof size without salts. Confidence: moderate on timing; one shared machine and three measurements. Compare within each table; timing distributions differed between phases. Unsalted Merkle privacy remains unproved.

Baseline parameters were increased until their global query-error contribution was below $2^{-129}$; other baseline terms retain v1's existing analysis. Candidate accounting includes folding, batching, initial OOD identification, and sampler bias. These are matched query margins, not identical end-to-end security certificates.

## Reproduce

Base: `0ce22913a9b4389eae08c9fd948008bfe7b01889`. Raw measurements and conditional accounting: `/private/tmp/whir-v1-prefix-unsalted-results/{w1,w2,paired}/{measurements.jsonl,summary.json}`. The paired driver is `/private/tmp/whir-v1-prefix-unsalted-results/paired_benchmark.py`; the saved reference binary is `/private/tmp/whir-v1-prefix-age-results/prefix-experiment-salted`. Fixture counts: Provekit `.claude/complete-age-check-performance-2026-10-22/v1_sha256/prepare.txt:13`; outer mask placement: `4ee40639fb8849aeeba37761fdda07f28367e81d`, `provekit/prover/src/whir_r1cs.rs`, `commit` method.

```sh
cargo build --offline --release --features experimental-prefix-zk --bin prefix-experiment
PYTHONDONTWRITEBYTECODE=1 python3 .claude/whir-zk-benchmark-2026-11-04.py target/release/prefix-experiment /private/tmp/whir-zk-results-w1 --witness-size 657980
PYTHONDONTWRITEBYTECODE=1 python3 .claude/whir-zk-benchmark-2026-11-04.py target/release/prefix-experiment /private/tmp/whir-zk-results-w2 --witness-size 655177
cargo test --offline --features experimental-prefix-zk --lib
cargo test --offline --release --features experimental-prefix-zk --lib whir_prefix
cargo test --offline --no-default-features --features experimental-prefix-zk --lib whir_prefix
cargo test --offline --release --all-features --lib whir_prefix
cargo clippy --offline --features experimental-prefix-zk --lib --bin prefix-experiment -- -D warnings -A clippy::assert_is_empty
```

Validation: 165 library tests passed, 10 existing tests ignored; 17 experiment tests passed in release. Nightly Rustfmt and all-target/all-feature Clippy passed with baseline lint allowances. The Clippy allowance covers existing assertions outside this patch. Rejection tests run in normal release mode; `verifier_panics` changes rejection into a panic, so all-feature runs exercise positive and arithmetic tests.

Strict all-target Clippy also flags existing code in `benches/zk_whir.rs` and `src/transcript/mock_sponge.rs`. With those baseline lints allowed, all-target/all-feature Clippy passes; unrelated files remain unchanged.
