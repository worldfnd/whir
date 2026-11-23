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
  -l, --security-level <SECURITY_LEVEL>          [default: 128]
  -p, --pow-bits <POW_BITS>                      [default: 20]
  -d, --num-variables <NUM_VARIABLES>            [default: 20]
  -e, --evaluations <NUM_EVALUATIONS>            [default: 1]
      --linear-constraints <NUM_LINEAR_CONSTRAINTS>  [default: 0]
  -r, --rate <RATE>                              [default: 1]
      --reps <VERIFIER_REPETITIONS>              [default: 1000]
  -i, --initfold <FIRST_ROUND_FOLDING_FACTOR>    [default: 4]
  -k, --fold <FOLDING_FACTOR>                    [default: 4]
      --decoding-regime <DECODING_REGIME>        [default: Johnson]
  -f, --field <FIELD>                            [default: Goldilocks3]
      --hash <HASH>                              [default: Blake3]
      --schedule <SCHEDULE>                      [default: stepping]
      --zk
  -h, --help                                     Print help
  -V, --version                                  Print version
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
