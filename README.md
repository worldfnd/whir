<h1 align="center">WHIR 🌪️</h1>

This library was developed using the [arkworks](https://arkworks.rs) ecosystem to accompany [WHIR 🌪️](https://eprint.iacr.org/2024/1586). 
By [Gal Arnon](https://galarnon42.github.io/) [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Giacomo Fenzi](https://gfenzi.io), and [Eylon Yogev](https://www.eylonyogev.com/about).

**WARNING:** This is an academic prototype and has not received careful code review. This implementation is NOT ready for production use.

<p align="center">
    <a href="https://github.com/WizardOfMenlo/whir/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/WizardOfMenlo/whir/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

# Usage
```
cargo run --release -- --help

Usage: main [OPTIONS]

Options:
  -t, --type <PROTOCOL_TYPE>             [default: PCS]
  -l, --security-level <SECURITY_LEVEL>  [default: 100]
  -p, --pow-bits <POW_BITS>
  -d, --num-variables <NUM_VARIABLES>    [default: 20]
  -e, --evaluations <NUM_EVALUATIONS>    [default: 1]
  -r, --rate <RATE>                      [default: 1]
      --reps <VERIFIER_REPETITIONS>      [default: 1000]
  -k, --fold <FOLDING_FACTOR>            [default: 4]
      --sec <SOUNDNESS_TYPE>             [default: ConjectureList]
      --fold_type <FOLD_OPTIMISATION>    [default: ProverHelps]
  -f, --field <FIELD>                    [default: Goldilocks2]
      --hash <MERKLE_TREE>               [default: Blake3]
  -h, --help                             Print help
  -V, --version                          Print version
```

Options:
- `-t` can be either `PCS` or `LDT` to run as a (multilinear) PCS or a LDT
- `-l` sets the (overall) security level of the scheme
- `-p` sets the number of PoW bits (used for the query-phase). PoW bits for proximity gaps are set automatically.
- `-d` sets the number of variables of the scheme.
- `-e` sets the number of evaluations to prove. Only meaningful in PCS mode.
- `-r` sets the log_inv of the rate
- `-k` sets the number of variables to fold at each iteration. 
- `--sec` sets the settings used to compute security. Available `UniqueDecoding`, `ProvableList`, `ConjectureList`
- `--fold_type` sets the settings used to compute folds. Available `Naive`, `ProverHelps`
- `-f` sets the field used, available are `Goldilocks2, Goldilocks3, Field192, Field256`.
- `--hash` sets the hash used for the Merkle tree, available are `SHA3` and `Blake3`

# Benchmarking Zook

`tests/zk_bench.rs` times Zook commit, prove and verify. It is an ignored test, so it only runs when asked for:

```
LOG=20 MODE=std ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
```

Each run prints one line per mode with the best time over `ITERS` runs, for example:

```
RESULT log=20 mode=Standard commit=0.170s prove=0.557s verify=0.0009s proof_digest=919fee7ff999ec11
```

`proof_digest` is a hash of the Standard-mode proof. It is deterministic, so two builds that should produce identical proofs must print the same digest. ZK proofs are randomised, so ZK runs print `-` instead.

Settings are read from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LOG` | `20` | log2 of the witness length |
| `MODE` | `std,zk` | comma-separated modes: `std`, `zk` |
| `ITERS` | `3` | runs per mode; the best time is reported |
| `SEC` | `128` | target security bits |
| `POW` | `10` | proof-of-work bits per step |
| `RATE` | `2` | starting log2 inverse rate (`2` = rate 1/4) |
| `FOLD` | `4` | folding factor |
| `HASH` | `sha2` | Merkle hash: `sha2` or `blake3` |
| `EMB` | `base` | `base`: Goldilocks witness, cubic-extension claims; `ext`: cubic-extension witness |

With the cubic Goldilocks extension, ZK mode cannot reach 128 bits at larger sizes, so lower `SEC` for ZK runs (about 112 bits is the most that works at 2^20, and 99 at 2^22):

```
LOG=20 MODE=std        ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
LOG=20 MODE=zk SEC=100 ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
LOG=22 MODE=std        ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
LOG=22 MODE=zk SEC=99  ITERS=3 cargo test --release --test zk_bench -- --ignored --nocapture
```

To compare two versions, run the same commands on each, then compare the times and check that the Standard-mode `proof_digest` values match.
