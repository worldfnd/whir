# Buffer Abstraction Design

## Purpose

The prover manipulates large vectors, encoded Reed-Solomon matrices, and Merkle
node arrays. On CPU, those objects can be ordinary `Vec`s. On an accelerator,
asking for a slice of the same object can force a synchronization and a full
readback.

The abstraction exists to make ownership explicit:

```text
Large prover data lives in an active backend object.
Protocol code only reads proof-sized data.
```

Proof-sized data means transcript scalars, Merkle roots, selected rows, selected
authentication nodes, and final folded vectors that are sent directly into the
proof. Full vectors, full encoded matrices, and full Merkle trees should stay in
`ActiveBuffer` or `ActiveMatrix`.

## Design Rules

The buffer layer is a data and math boundary. It should not own protocol logic.

```text
Protocol layer:
  transcript order
  challenge sampling
  proof layout
  round structure
  hash choice
  verification semantics

Buffer layer:
  backend-owned vectors
  backend-owned matrices
  full-buffer math loops
  selected reads
  explicit full reads
```

That split rules out protocol-specific buffer APIs. WHIR should keep one prove
path, IRS should keep one commit path, and buffers should not write to the
transcript. Each protocol should place backend-owned data only at the boundary
where that protocol actually consumes large resident data.

## Active Types

Protocol code refers to active aliases instead of hard-coding a backend:

```rust
pub type ActiveBuffer<T> = selected_backend::Buffer<T>;
pub type ActiveMatrix<T> = selected_backend::Matrix<T>;
```

The current selected backend is CPU, but that is a local implementation detail
inside the buffer module. The aliases are intentionally central. WHIR, IRS,
sumcheck, matrix commitment, and Merkle opening operate on `ActiveBuffer` and
`ActiveMatrix`, so a later backend can be selected at this boundary without
changing protocol callsites.

The active types also expose the explicit construction and read boundaries used
by protocol code:

```rust
// Setup or protocol-owned construction.
ActiveBuffer::from_slice(...);
ActiveBuffer::from_vec(...);
ActiveBuffer::zeros(...);
ActiveBuffer::random(...);
ActiveMatrix::from_vec(...);

// Proof-boundary materialization.
buffer.read();
buffer.read_index(...);
buffer.read_indices(...);
matrix.read_rows(...);
```

These are intentionally not part of `BufferOps`. They are ownership and
materialization boundaries, while `BufferOps` is only the full-buffer math
surface.

## Core Trait

`BufferOps` covers operations that are naturally full-buffer operations. Loops
over protocol rounds remain outside the trait.

```rust
pub trait BufferOps<F: Field>: Clone {
    // Same backend family for another field.
    type Buffer<T: Field>: BufferOps<T>;

    // Metadata.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;

    // In-place full-buffer transforms.
    fn fold(&mut self, weight: F);

    // Full-buffer reductions used by sumcheck and WHIR.
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F);
    fn fold_and_sumcheck_polynomial(
        &mut self,
        other: &mut Self,
        weight: F,
    ) -> (F, F);

    // Mixed-field full-buffer reductions used by protocol checks.
    fn mixed_dot<T: Field, M: Embedding<Source = F, Target = T>>(
        &self,
        embedding: &M,
        other: &Self::Buffer<T>,
    ) -> M::Target;

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target;

    // Writes into backend-owned state instead of returning a large vector.
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::Buffer<M::Target>,
        weight: M::Target,
    );

    fn geometric_accumulate(&mut self, scalars: &[F], points: &[F]);
}
```

Construction and host materialization are explicit active-type boundaries, not
part of this math trait. Protocol code may upload data at commitment setup and
may read proof-sized data at transcript/opening boundaries, but those actions
should stay visible at the call site.

Scalar-returning methods are still synchronization points. That is acceptable
when the scalar is transcript data. If a later backend kernel should consume the
result directly, the right design is a fused operation or a backend-owned scalar,
not a hidden readback followed by an upload.

## Matrix Shape

`ActiveMatrix<T>` is the backend-owned row matrix produced by interleaved RS
encoding and consumed by matrix commitment.

The protocol-visible matrix surface is intentionally small: shape metadata and
selected logical row readback.

The caller owns the protocol metadata: row count, column count, query indices,
and how selected rows are written into the proof. The backend owns physical
layout. For example, an accelerated matrix may store rows in a different order
or column-major internally, but `read_rows(indices)` returns logical rows.

## IRS Boundary

IRS remains protocol code. It samples masks, calls the RS encoder, commits the
encoded rows, samples OOD points, and records OOD evaluations.

The shared RS primitive is:

```rust
pub fn interleaved_rs_encode<F: Field + 'static>(
    vectors: &[&ActiveBuffer<F>],
    masks: &ActiveBuffer<F>,
    message_length: usize,
    interleaving_depth: usize,
    codeword_length: usize,
) -> ActiveMatrix<F>;
```

This replaces the old call shape that required `&[&[F]]` and a mask slice. IRS
does not need to expose host slices just to encode.

IRS witness names describe what is actually stored:

```rust
pub struct Witness<F: Field, G = F>
where
    G: Field,
{
    pub masks: ActiveBuffer<F>,
    pub encoded_matrix: ActiveMatrix<F>,
    pub encoded_matrix_witness: matrix_commit::Witness,
    pub out_of_domain: Evaluations<G>,
}
```

The masks and encoded matrix remain backend-owned. OOD evaluations are still
host vectors because they are proof/transcript data in the current protocol.

## Matrix Commitment Boundary

Matrix commitment owns the hash choice and transcript write. The buffer layer
does not commit rows and does not select a hash function.

The commit path is:

```text
ActiveMatrix<T>
  -> matrix_commit::Config::build_nodes(...)
  -> ActiveBuffer<Hash>
  -> merkle_tree::Config::commit_nodes(...)
  -> root written to transcript
```

The important ownership change is that the Merkle witness stores the node array
as an active hash buffer:

```rust
pub struct Witness {
    nodes: ActiveBuffer<Hash>,
}
```

Opening a Merkle tree reads only the required authentication nodes. That keeps
the full tree resident and turns openings into proof-sized reads.

## WHIR Prove Boundary

WHIR `prove` accepts active buffers for the committed vectors and borrowed
witness references:

```rust
pub fn prove<'a, H, R>(
    &self,
    prover_state: &mut ProverState<H, R>,
    vectors: &[&ActiveBuffer<M::Source>],
    witnesses: &'a [&'a Witness<M::Target, M>],
    linear_forms: Vec<Box<dyn LinearForm<M::Target>>>,
    evaluations: Cow<'a, [M::Target]>,
) -> FinalClaim<M::Target>
```

The WHIR prover still owns the protocol decisions:

```text
sample vector RLC coefficients
build a backend-owned linear combination
sample constraint RLC coefficients
build the covector from linear forms
add OOD and STIR constraints
run sumcheck rounds
commit and open IRS rounds
send the final folded vector
```

The buffer layer only performs the large vector work inside those steps:

```rust
vector.mixed_scalar_mul_add_to(...);
covector.geometric_accumulate(...);
sumcheck.prove(..., &mut vector, &mut covector, ...);
```

## Readback Policy

Expected readbacks are proof-sized: Merkle roots, authentication nodes, selected
IRS rows, transcript scalars, and the final folded WHIR vector. Suspicious
readbacks are full committed vectors, full encoded matrices, full Merkle trees,
or a readback immediately followed by upload into another backend buffer.

The current CPU implementation may borrow slices inside CPU-only primitives,
but that access stays crate-private. IRS, WHIR, and matrix commitment callers
should see active buffers and explicit proof-boundary reads.

## Open Work

Scalar-returning reductions currently return host scalars. That is fine for
transcript values, but later backend work may need fused operations or
backend-owned scalar handles when the scalar feeds another backend computation.

The ZK wrapper remains a host-side wrapper for now. Its outer API accepts host
slices/owned vectors, builds masked and blinding vectors on the host, and only
constructs active buffers at the point where it calls the inner WHIR
commit/prove paths. That keeps this refactor focused on the regular WHIR
residency boundary without pushing buffer requirements through unfinished ZK
code.

Future backend selection should happen behind `ActiveBuffer` and `ActiveMatrix`.
Protocol APIs should not grow backend-specific entrypoints.
