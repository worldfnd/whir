<h1 align="center">WHIR 🌪️</h1>

<p align="center">
    <a href="https://github.com/worldfnd/whir/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-APACHE-blue.svg"></a>
    <a href="https://github.com/worldfnd/whir/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

A Rust implementation of the [WHIR](https://eprint.iacr.org/2024/1586) polynomial commitment scheme, with a standard mode and a zero-knowledge mode.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [Library](#library)
  - [Command line](#command-line)
- [Crate layout](#crate-layout)
- [Contributing](#contributing)
- [License](#license)

## Background

WHIR is an interactive oracle proof of proximity for constrained Reed–Solomon codes. Compiled with Merkle commitments and Fiat–Shamir, it gives a polynomial commitment scheme: the prover commits to a vector and proves claimed values of linear forms on it, such as multilinear evaluations. WHIR is by [Gal Arnon](https://galarnon42.github.io/), [Alessandro Chiesa](https://ic-people.epfl.ch/~achiesa/), [Giacomo Fenzi](https://gfenzi.io), and [Eylon Yogev](https://www.eylonyogev.com/about).

This crate runs one protocol in two modes:

- `Mode::Standard`: Unique, Johnson, or Capacity decoding.
- `Mode::ZeroKnowledge`: masks the witness so the proof reveals nothing beyond the claimed values. Requires Johnson decoding.

`JointOpeningConfig` opens two separately committed vectors with one proof.

Parameters and soundness bounds follow the [ProveKit whitepaper](https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper).

## Install

Releases are published on crates.io as [`provekit-whir`](https://crates.io/crates/provekit-whir). The next release will carry the API this README documents; until then, depend on `main`. Use one of:

```toml
[dependencies]
# Latest release
whir = { package = "provekit-whir", version = "0.2" }

# Current main (the API this README documents)
whir = { git = "https://github.com/worldfnd/whir" }
```

Cargo features:

- `parallel` (default): multithreaded prover through rayon.
- `asm`: assembly field arithmetic from `ark-ff`.
- `tracing`: tracing spans for commit, prove, and verify.
- `verifier_panics`: the verifier panics instead of returning an error. For debugging.

Minimum supported Rust version: 1.88.

## Usage

### Library

Commit to a vector over Goldilocks, prove one multilinear evaluation over its cubic extension, and verify it:

```rust
use whir::{
    algebra::{
        embedding::Basefield,
        fields::{Field64, Field64_3},
        linear_form::{Evaluate, LinearForm, MultilinearExtension},
    },
    buffer::Buffer,
    hash,
    transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    DecodingRegime, Mode, Params, ProtocolConfig, RateSchedule,
};

// Witness in Field64; linear forms and challenges in Field64_3.
type M = Basefield<Field64_3>;

let config = ProtocolConfig::<M>::derive(Params {
    mode: Mode::Standard,
    regime: DecodingRegime::Johnson,
    bits: 128,        // security level
    pow: 20,          // maximum proof-of-work bits per step
    hash: hash::BLAKE3,
    size: 1 << 16,    // vector length
    rate: 1,          // log inverse of the starting rate
    folds: [4, 4],    // log folding factor: first round, later rounds
    schedule: RateSchedule::Stepping,
})
.expect("valid parameters");

let witness: Vec<Field64> = (0..1u64 << 16).map(Field64::from).collect();
let form = MultilinearExtension::<Field64_3> {
    point: vec![Field64_3::from(7u64); 16],
};
let value = form.evaluate(&M::default(), &witness);
let forms: [&dyn LinearForm<Field64_3>; 1] = [&form];

// The protocol id binds the transcript to the configuration.
let ds = DomainSeparator::protocol(&config)
    .session(&"readme example")
    .instance(&Empty);

// Prover
let mut ps = ProverState::new_std(&ds);
let committed = config.commit(&mut ps, Buffer::from(witness));
config.prove(&mut ps, committed, &forms, &[value]);
let proof = ps.proof();

// Verifier
let mut vs = VerifierState::new_std(&ds, &proof);
let commitment = config.receive_commitment(&mut vs).unwrap();
let final_claim = config.verify(&mut vs, commitment, &forms, &[value]).unwrap();
// The verifier defers evaluating the linear forms; check them last.
final_claim.verify(&forms).unwrap();
```

Set `mode: Mode::ZeroKnowledge` for the zero-knowledge mode.

### Command line

`main` prints the derived round parameters, commits, proves, and verifies once, then prints prover time, proof size, average verifier time, and hash count:

```
cargo run --release -- --help

Usage: main [OPTIONS]

Options:
  -l, --security-level <SECURITY_LEVEL>
          [default: 128]
  -p, --pow-bits <POW_BITS>
          Maximum proof of work difficulty in bits [default: 20]
  -d, --num-variables <NUM_VARIABLES>
          [default: 20]
  -e, --evaluations <NUM_EVALUATIONS>
          [default: 1]
      --linear-constraints <NUM_LINEAR_CONSTRAINTS>
          [default: 0]
  -r, --rate <RATE>
          [default: 1]
      --reps <VERIFIER_REPETITIONS>
          [default: 1000]
  -i, --initfold <FIRST_ROUND_FOLDING_FACTOR>
          [default: 4]
  -k, --fold <FOLDING_FACTOR>
          [default: 4]
      --decoding-regime <DECODING_REGIME>
          Reed–Solomon decoding regime: Unique, Johnson, or Capacity [default: Johnson]
  -f, --field <FIELD>
          [default: Goldilocks3]
      --hash <HASH>
          [default: Blake3]
      --schedule <SCHEDULE>
          Rate schedule: `stepping`, `capped:<max log inv rate>`, or `adaptive:<knee weight>` [default: adaptive:0.5]
      --zk
          Run in zero-knowledge mode. Requires Johnson decoding
  -h, --help
          Print help
  -V, --version
          Print version
```

Options:

- `-l` sets the security level in bits.
- `-p` sets the maximum proof-of-work bits per protocol step.
- `-d` sets the number of variables. The committed vector has `2^d` entries.
- `-e` sets the number of evaluation claims to prove.
- `--linear-constraints` sets the number of extra linear-form claims to prove. `-e` plus `--linear-constraints` must be at least 1.
- `-r` sets the log inverse of the starting rate.
- `-i` sets the log folding factor of the first round. `-k` sets it for all later rounds.
- `--decoding-regime` selects the soundness bound: `Unique`, `Johnson`, or `Capacity`.
- `-f` sets the field: `Goldilocks1`, `Goldilocks2`, `Goldilocks3`, `Field128`, `Field192`, `Field256`.
- `--hash` sets the Merkle tree hash: `Sha2`, `Sha3`, `Keccak`, `Blake3`.
- `--schedule` sets the rate schedule: `stepping`, `capped:<max log inverse rate>`, or `adaptive:<knee weight in [0,1]>`.
- `--zk` runs the zero-knowledge mode. It requires `Johnson` decoding.

## Crate layout

The crate documentation (`cargo doc --open`) has a full file map. In short:

| Path | Contents |
| --- | --- |
| `src/whir/params/` | `Params` → `ProtocolConfig`: round construction and soundness bits |
| `src/whir/` | commit, prove, verify, the round loop, zero-knowledge masks, joint openings |
| `src/protocols/` | sub-protocols: sumcheck, code switching, IRS commitment, Merkle trees, proof of work |
| `src/decoding.rs` | Reed–Solomon decoding regimes and bounds |
| `src/algebra/` | fields, embeddings, linear forms, NTT |
| `src/transcript/` | Fiat–Shamir transcript |

## Contributing

Open an issue or pull request on [GitHub](https://github.com/worldfnd/whir). Before you push, run:

```sh
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -Dwarnings
cargo test --release --all-features --all-targets
```

CI runs these checks, plus typos and unused-dependency checks.

## License

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
