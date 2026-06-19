// NOTE: 100% AI GENERATED

use std::{
    any::type_name,
    cell::OnceCell,
    collections::HashMap,
    marker::PhantomData,
    os::raw::c_void,
    sync::{Mutex, OnceLock},
    time::Instant,
};

use ark_ff::{AdditiveGroup, BigInt, FftField, Field, Fp, MontBackend};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use zerocopy::IntoBytes;

use crate::{
    algebra::fields::{BN254Config, Field256},
    buffer::{BufferOps, MetalBuffer},
    hash::Hash as Digest,
};

use super::buffer::MetalFieldBuffer;
use super::kernels::{
    GEOMETRIC_ACCUMULATE_POINT_BLOCK_BATCH_BYTES, GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS,
    GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE, GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD,
    LARGE_GEOMETRIC_ACCUMULATE_CHUNK_SIZE, LARGE_GEOMETRIC_ACCUMULATE_THRESHOLD,
    MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE, METAL_SOURCE, REDUCTION_CHUNK_SIZE,
    SMALL_GEOMETRIC_ACCUMULATE_CHUNK_SIZE,
};
use super::profile;

pub(crate) struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    fold: ComputePipelineState,
    fold_pair: ComputePipelineState,
    scalar_mul_add: ComputePipelineState,
    scalar_mul: ComputePipelineState,
    dot: ComputePipelineState,
    sumcheck: ComputePipelineState,
    dot_chunks: ComputePipelineState,
    sum_chunks: ComputePipelineState,
    sumcheck_reduce_chunks: ComputePipelineState,
    sumcheck_chunks: ComputePipelineState,
    fold_pair_sumcheck_chunks: ComputePipelineState,
    geometric_accumulate: ComputePipelineState,
    geometric_accumulate_chunks: ComputePipelineState,
    geometric_accumulate_chunks_strided: ComputePipelineState,
    geometric_accumulate_point_blocks: ComputePipelineState,
    geometric_accumulate_point_blocks_range: ComputePipelineState,
    geometric_accumulate_reduce_point_blocks: ComputePipelineState,
    univariate_evaluate: ComputePipelineState,
    univariate_eval_chunks: ComputePipelineState,
    interleaved_rs_encode: ComputePipelineState,
    interleaved_rs_encode_single_vector: ComputePipelineState,
    multilinear_extend: ComputePipelineState,
    pack_single_vector_cosets: ComputePipelineState,
    apply_coset_twiddles: ComputePipelineState,
    replicate_first_coset: ComputePipelineState,
    bit_reverse_rows: ComputePipelineState,
    ntt_stage_rows: ComputePipelineState,
    transpose: ComputePipelineState,
    transpose_reverse_rows: ComputePipelineState,
    encode_field_rows_le: ComputePipelineState,
    read_rows: ComputePipelineState,
    ntt_roots: Mutex<HashMap<usize, MetalFieldBuffer>>,
    root_powers: Mutex<HashMap<usize, MetalFieldBuffer>>,
}

pub fn init() {
    let _ = runtime();
}

pub(crate) fn runtime() -> &'static MetalRuntime {
    static RUNTIME: OnceLock<MetalRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let device = Device::system_default().expect("Metal device is not available");
        let library = device
            .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
            .expect("failed to compile Metal BN254 kernels");
        let pipeline = |name: &str| {
            let function = library
                .get_function(name, None)
                .unwrap_or_else(|_| panic!("missing Metal kernel {name}"));
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap_or_else(|err| panic!("failed to compile Metal kernel {name}: {err}"))
        };
        MetalRuntime {
            queue: device.new_command_queue(),
            fold: pipeline("bn254_fold"),
            fold_pair: pipeline("bn254_fold_pair"),
            scalar_mul_add: pipeline("bn254_scalar_mul_add"),
            scalar_mul: pipeline("bn254_scalar_mul"),
            dot: pipeline("bn254_dot"),
            sumcheck: pipeline("bn254_sumcheck"),
            dot_chunks: pipeline("bn254_dot_chunks"),
            sum_chunks: pipeline("bn254_sum_chunks"),
            sumcheck_reduce_chunks: pipeline("bn254_sumcheck_reduce_chunks"),
            sumcheck_chunks: pipeline("bn254_sumcheck_chunks"),
            fold_pair_sumcheck_chunks: pipeline("bn254_fold_pair_sumcheck_chunks"),
            geometric_accumulate: pipeline("bn254_geometric_accumulate"),
            geometric_accumulate_chunks: pipeline("bn254_geometric_accumulate_chunks"),
            geometric_accumulate_chunks_strided: pipeline(
                "bn254_geometric_accumulate_chunks_strided",
            ),
            geometric_accumulate_point_blocks: pipeline("bn254_geometric_accumulate_point_blocks"),
            geometric_accumulate_point_blocks_range: pipeline(
                "bn254_geometric_accumulate_point_blocks_range",
            ),
            geometric_accumulate_reduce_point_blocks: pipeline(
                "bn254_geometric_accumulate_reduce_point_blocks",
            ),
            univariate_evaluate: pipeline("bn254_univariate_evaluate"),
            univariate_eval_chunks: pipeline("bn254_univariate_eval_chunks"),
            interleaved_rs_encode: pipeline("bn254_interleaved_rs_encode"),
            interleaved_rs_encode_single_vector: pipeline(
                "bn254_interleaved_rs_encode_single_vector",
            ),
            multilinear_extend: pipeline("bn254_multilinear_extend"),
            pack_single_vector_cosets: pipeline("bn254_pack_single_vector_cosets"),
            apply_coset_twiddles: pipeline("bn254_apply_coset_twiddles"),
            replicate_first_coset: pipeline("bn254_replicate_first_coset"),
            bit_reverse_rows: pipeline("bn254_bit_reverse_permute_rows_in_place"),
            ntt_stage_rows: pipeline("bn254_radix2_ntt_stage_rows_in_place"),
            transpose: pipeline("bn254_transpose_matrix"),
            transpose_reverse_rows: pipeline("bn254_transpose_matrix_reverse_rows"),
            encode_field_rows_le: pipeline("bn254_encode_field_rows_le"),
            read_rows: pipeline("bn254_read_rows"),
            ntt_roots: Mutex::new(HashMap::new()),
            root_powers: Mutex::new(HashMap::new()),
            device,
        }
    })
}

pub(crate) fn pipeline<'a>(rt: &'a MetalRuntime, name: &str) -> &'a ComputePipelineState {
    match name {
        "bn254_fold" => &rt.fold,
        "bn254_fold_pair" => &rt.fold_pair,
        "bn254_scalar_mul_add" => &rt.scalar_mul_add,
        "bn254_scalar_mul" => &rt.scalar_mul,
        "bn254_dot" => &rt.dot,
        "bn254_sumcheck" => &rt.sumcheck,
        "bn254_dot_chunks" => &rt.dot_chunks,
        "bn254_sum_chunks" => &rt.sum_chunks,
        "bn254_sumcheck_reduce_chunks" => &rt.sumcheck_reduce_chunks,
        "bn254_sumcheck_chunks" => &rt.sumcheck_chunks,
        "bn254_fold_pair_sumcheck_chunks" => &rt.fold_pair_sumcheck_chunks,
        "bn254_geometric_accumulate" => &rt.geometric_accumulate,
        "bn254_geometric_accumulate_chunks" => &rt.geometric_accumulate_chunks,
        "bn254_geometric_accumulate_chunks_strided" => &rt.geometric_accumulate_chunks_strided,
        "bn254_geometric_accumulate_point_blocks" => &rt.geometric_accumulate_point_blocks,
        "bn254_geometric_accumulate_point_blocks_range" => {
            &rt.geometric_accumulate_point_blocks_range
        }
        "bn254_geometric_accumulate_reduce_point_blocks" => {
            &rt.geometric_accumulate_reduce_point_blocks
        }
        "bn254_univariate_evaluate" => &rt.univariate_evaluate,
        "bn254_univariate_eval_chunks" => &rt.univariate_eval_chunks,
        "bn254_interleaved_rs_encode" => &rt.interleaved_rs_encode,
        "bn254_interleaved_rs_encode_single_vector" => &rt.interleaved_rs_encode_single_vector,
        "bn254_multilinear_extend" => &rt.multilinear_extend,
        "bn254_pack_single_vector_cosets" => &rt.pack_single_vector_cosets,
        "bn254_apply_coset_twiddles" => &rt.apply_coset_twiddles,
        "bn254_replicate_first_coset" => &rt.replicate_first_coset,
        "bn254_bit_reverse_permute_rows_in_place" => &rt.bit_reverse_rows,
        "bn254_radix2_ntt_stage_rows_in_place" => &rt.ntt_stage_rows,
        "bn254_transpose_matrix" => &rt.transpose,
        "bn254_transpose_matrix_reverse_rows" => &rt.transpose_reverse_rows,
        "bn254_encode_field_rows_le" => &rt.encode_field_rows_le,
        "bn254_read_rows" => &rt.read_rows,
        _ => panic!("unknown Metal kernel {name}"),
    }
}

pub(crate) fn new_shared_buffer(rt: &MetalRuntime, bytes: u64) -> Buffer {
    profile::record_alloc(bytes);
    let buffer = rt
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    profile::record_device_allocated(rt.device.current_allocated_size());
    buffer
}

pub(crate) fn new_shared_buffer_with_data(
    rt: &MetalRuntime,
    data: *const c_void,
    bytes: u64,
) -> Buffer {
    let start = Instant::now();
    let buffer = rt
        .device
        .new_buffer_with_data(data, bytes, MTLResourceOptions::StorageModeShared);
    profile::record_alloc(bytes);
    profile::record_upload(bytes, start.elapsed());
    profile::record_device_allocated(rt.device.current_allocated_size());
    buffer
}

pub(crate) fn wait_for_command_named(command: &metal::CommandBufferRef, label: &str) {
    command.commit();
    let start = Instant::now();
    command.wait_until_completed();
    let elapsed = start.elapsed();
    if std::env::var_os("WHIR_METAL_TRACE").is_some() {
        eprintln!(
            "metal command {label} {:.3} ms",
            elapsed.as_secs_f64() * 1_000.0
        );
    }
    profile::record_command_wait(elapsed);
}

pub(crate) fn wait_for_blit(command: &metal::CommandBufferRef, bytes: u64) {
    command.commit();
    let start = Instant::now();
    command.wait_until_completed();
    profile::record_blit(bytes, start.elapsed());
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BitReverseParams {
    row_len: u32,
    log_n: u32,
    total_elements: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NttStageParams {
    row_len: u32,
    half_m: u32,
    twiddle_offset: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TransposeParams {
    rows: u32,
    cols: u32,
    total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReplicateCosetsParams {
    row_len: u32,
    coset_size: u32,
    trailing_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackSingleVectorParams {
    row_count: u32,
    message_length: u32,
    codeword_length: u32,
    coset_size: u32,
    total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FieldBytesParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ApplyCosetTwiddlesParams {
    row_count: u32,
    num_cosets: u32,
    coset_size: u32,
    codeword_length: u32,
    total_elements: u32,
}

pub(crate) fn parallel_dot(a: &MetalFieldBuffer, b: &MetalFieldBuffer, len: usize) -> Field256 {
    parallel_dot_at(a, 0, b, 0, len)
}

/// Inner product of `a[a_off..a_off+len]` and `b[b_off..b_off+len]`.
pub(crate) fn parallel_dot_at(
    a: &MetalFieldBuffer,
    a_off: usize,
    b: &MetalFieldBuffer,
    b_off: usize,
    len: usize,
) -> Field256 {
    if len == 0 {
        return Field256::ZERO;
    }
    let partial_count = len.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel_with_offsets(
        command,
        pipeline(rt, "bn254_dot_chunks"),
        &[&a.limbs, &b.limbs, &partials.limbs],
        &[field_byte_offset(a_off), field_byte_offset(b_off), 0],
        &[len as u32, REDUCTION_CHUNK_SIZE as u32],
        partial_count,
    );
    let (result, offset) = encode_field_reduction(command, partials, partial_count, 0);
    wait_for_command_named(command, "bn254_dot_chunks");
    download_field_at(&result.limbs, offset)
}

pub(crate) fn parallel_sumcheck(
    a: &MetalFieldBuffer,
    b: &MetalFieldBuffer,
    len: usize,
    fold_half: usize,
) -> (Field256, Field256) {
    parallel_sumcheck_at(a, 0, b, 0, len, fold_half)
}

/// Sumcheck `(c0, c2)` over `a[a_off..]` and `b[b_off..]` (logical length `len`).
pub(crate) fn parallel_sumcheck_at(
    a: &MetalFieldBuffer,
    a_off: usize,
    b: &MetalFieldBuffer,
    b_off: usize,
    len: usize,
    fold_half: usize,
) -> (Field256, Field256) {
    let partial_count = fold_half.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * 2 * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel_with_offsets(
        command,
        pipeline(rt, "bn254_sumcheck_chunks"),
        &[&a.limbs, &b.limbs, &partials.limbs],
        &[field_byte_offset(a_off), field_byte_offset(b_off), 0],
        &[
            len as u32,
            fold_half as u32,
            REDUCTION_CHUNK_SIZE as u32,
            partial_count as u32,
        ],
        partial_count,
    );
    let result = encode_sumcheck_reduction(command, partials, partial_count);
    wait_for_command_named(command, "bn254_sumcheck_chunks");
    download_sumcheck_pair(&result)
}

pub(crate) fn parallel_fold_pair_sumcheck(
    a: &MetalFieldBuffer,
    b: &MetalFieldBuffer,
    weight: &MetalFieldBuffer,
    len: usize,
    fold_half: usize,
) -> (Field256, Field256) {
    let sum_half = fold_half.next_power_of_two() >> 1;
    debug_assert!(sum_half > 0);
    let partial_count = sum_half.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * 2 * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(
        command,
        pipeline(rt, "bn254_fold_pair_sumcheck_chunks"),
        &[&a.limbs, &b.limbs, &weight.limbs, &partials.limbs],
        &[
            len as u32,
            fold_half as u32,
            sum_half as u32,
            REDUCTION_CHUNK_SIZE as u32,
            partial_count as u32,
        ],
        partial_count,
    );
    let result = encode_sumcheck_reduction(command, partials, partial_count);
    wait_for_command_named(command, "bn254_fold_pair_sumcheck_chunks");
    download_sumcheck_pair(&result)
}

pub(crate) fn parallel_univariate_evaluate(
    coeffs: &MetalFieldBuffer,
    point: &MetalFieldBuffer,
    len: usize,
) -> Field256 {
    parallel_univariate_evaluate_at(coeffs, 0, point, len)
}

/// Univariate evaluation of `coeffs[coeff_off..coeff_off+len]` at `point`.
pub(crate) fn parallel_univariate_evaluate_at(
    coeffs: &MetalFieldBuffer,
    coeff_off: usize,
    point: &MetalFieldBuffer,
    len: usize,
) -> Field256 {
    if len == 0 {
        return Field256::ZERO;
    }
    let partial_count = len.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel_with_offsets(
        command,
        pipeline(rt, "bn254_univariate_eval_chunks"),
        &[&coeffs.limbs, &point.limbs, &partials.limbs],
        &[field_byte_offset(coeff_off), 0, 0],
        &[len as u32, REDUCTION_CHUNK_SIZE as u32],
        partial_count,
    );
    let (result, offset) = encode_field_reduction(command, partials, partial_count, 0);
    wait_for_command_named(command, "bn254_univariate_eval_chunks");
    download_field_at(&result.limbs, offset)
}

pub(crate) fn parallel_geometric_accumulate_point_blocks(
    acc: &MetalFieldBuffer,
    points: &MetalFieldBuffer,
    point_steps: &MetalFieldBuffer,
    scalars: &MetalFieldBuffer,
    len: usize,
    num_points: usize,
    chunk_size: usize,
) {
    assert!(chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE);
    let point_blocks = num_points.div_ceil(GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE);
    let chunks = len.div_ceil(chunk_size);
    let partial_len = point_blocks
        .checked_mul(len)
        .expect("Metal geometric partial size overflow");
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_len * size_of::<Field256>()) as u64),
    };
    run_in_place(
        "bn254_geometric_accumulate_point_blocks",
        &[
            &partials.limbs,
            &points.limbs,
            &point_steps.limbs,
            &scalars.limbs,
        ],
        &[
            len as u32,
            num_points as u32,
            chunk_size as u32,
            GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE as u32,
            point_blocks as u32,
        ],
        chunks * point_blocks,
    );
    run_in_place(
        "bn254_geometric_accumulate_reduce_point_blocks",
        &[&acc.limbs, &partials.limbs],
        &[len as u32, point_blocks as u32],
        len,
    );
}

pub(crate) fn parallel_geometric_accumulate_point_blocks_batched(
    acc: &MetalFieldBuffer,
    points: &MetalFieldBuffer,
    point_steps: &MetalFieldBuffer,
    scalars: &MetalFieldBuffer,
    len: usize,
    num_points: usize,
    chunk_size: usize,
) {
    assert!(chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE);
    let point_blocks = num_points.div_ceil(GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE);
    let chunks = len.div_ceil(chunk_size);
    let bytes_per_point_block = len
        .checked_mul(size_of::<Field256>())
        .expect("Metal geometric partial size overflow");
    let default_batch_blocks =
        (GEOMETRIC_ACCUMULATE_POINT_BLOCK_BATCH_BYTES / bytes_per_point_block).max(1);
    let batch_blocks = std::env::var("WHIR_METAL_GEOM_POINT_BLOCK_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_batch_blocks)
        .min(point_blocks);
    let rt = runtime();
    for point_block_offset in (0..point_blocks).step_by(batch_blocks) {
        let current_batch = batch_blocks.min(point_blocks - point_block_offset);
        let partial_len = current_batch
            .checked_mul(len)
            .expect("Metal geometric partial size overflow");
        let partials = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (partial_len * size_of::<Field256>()) as u64),
        };
        run_in_place(
            "bn254_geometric_accumulate_point_blocks_range",
            &[
                &partials.limbs,
                &points.limbs,
                &point_steps.limbs,
                &scalars.limbs,
            ],
            &[
                len as u32,
                num_points as u32,
                chunk_size as u32,
                GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE as u32,
                point_block_offset as u32,
                current_batch as u32,
            ],
            chunks * current_batch,
        );
        run_in_place(
            "bn254_geometric_accumulate_reduce_point_blocks",
            &[&acc.limbs, &partials.limbs],
            &[len as u32, current_batch as u32],
            len,
        );
    }
}

pub(crate) fn geometric_accumulate_chunk_size(len: usize) -> usize {
    std::env::var("WHIR_METAL_GEOM_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(if len >= LARGE_GEOMETRIC_ACCUMULATE_THRESHOLD {
            LARGE_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        } else {
            SMALL_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        })
        .min(MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE)
}

pub(crate) fn should_use_geometric_point_blocks(
    len: usize,
    num_points: usize,
    chunk_size: usize,
) -> bool {
    chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        && num_points >= GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS
        && len <= GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD
}

pub(crate) fn should_use_geometric_point_blocks_batched(
    len: usize,
    num_points: usize,
    chunk_size: usize,
) -> bool {
    chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        && num_points >= GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS
        && len > GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD
}

/// Encodes all tree-reduction levels into `command` and returns the buffer
/// and offset holding the final scalar. Runs with a single command wait.
pub(crate) fn encode_field_reduction(
    command: &metal::CommandBufferRef,
    input: MetalFieldBuffer,
    len: usize,
    offset: usize,
) -> (MetalFieldBuffer, usize) {
    let rt = runtime();
    let mut current = input;
    let mut current_len = len;
    let mut current_offset = offset;
    while current_len > 1 {
        let next_len = current_len.div_ceil(REDUCTION_CHUNK_SIZE);
        let next = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (next_len * size_of::<Field256>()) as u64),
        };
        encode_u32_kernel(
            command,
            pipeline(rt, "bn254_sum_chunks"),
            &[&current.limbs, &next.limbs],
            &[
                current_len as u32,
                current_offset as u32,
                REDUCTION_CHUNK_SIZE as u32,
            ],
            next_len,
        );
        current = next;
        current_len = next_len;
        current_offset = 0;
    }
    (current, current_offset)
}

/// Encodes all (c0, c2) tree-reduction levels into `command` and returns the
/// buffer holding the final pair. Runs with a single command wait.
pub(crate) fn encode_sumcheck_reduction(
    command: &metal::CommandBufferRef,
    input: MetalFieldBuffer,
    len: usize,
) -> MetalFieldBuffer {
    let rt = runtime();
    let mut current = input;
    let mut current_len = len;
    while current_len > 1 {
        let next_len = current_len.div_ceil(REDUCTION_CHUNK_SIZE);
        let next = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (next_len * 2 * size_of::<Field256>()) as u64),
        };
        encode_u32_kernel(
            command,
            pipeline(rt, "bn254_sumcheck_reduce_chunks"),
            &[&current.limbs, &next.limbs],
            &[
                current_len as u32,
                REDUCTION_CHUNK_SIZE as u32,
                next_len as u32,
            ],
            next_len,
        );
        current = next;
        current_len = next_len;
    }
    current
}

pub(crate) fn download_sumcheck_pair(buffer: &MetalFieldBuffer) -> (Field256, Field256) {
    let values = download_field(&buffer.limbs, 2);
    (values[0], values[1])
}

pub(crate) fn encode_single_vector_coset_ntt<F: Field>(
    vector: &MetalBuffer<F>,
    message_length: usize,
    interleaving_depth: usize,
    codeword_length: usize,
    coset_size: usize,
) -> MetalBuffer<F> {
    assert!(codeword_length.is_power_of_two());
    assert!(Field256::get_root_of_unity(codeword_length as u64).is_some());
    assert_eq!(vector.len(), message_length * interleaving_depth);

    assert!(codeword_length.is_multiple_of(coset_size));
    let num_cosets = codeword_length / coset_size;
    let total_elements = interleaving_depth
        .checked_mul(codeword_length)
        .expect("Metal RS encode size overflow");
    assert!(total_elements <= u32::MAX as usize);

    let rt = runtime();
    let source = vector.bn254_buffer();
    let current = new_shared_buffer(rt, (total_elements * 4 * size_of::<u64>()) as u64);
    let transposed = new_shared_buffer(rt, (total_elements * 4 * size_of::<u64>()) as u64);
    let codeword_root_powers = root_powers_buffer(codeword_length);
    let coset_roots = roots_buffer(coset_size);

    let command = rt.queue.new_command_buffer();

    encode_kernel(
        &command,
        &rt.pack_single_vector_cosets,
        &[&source.limbs, &current],
        &PackSingleVectorParams {
            row_count: interleaving_depth as u32,
            message_length: message_length as u32,
            codeword_length: codeword_length as u32,
            coset_size: coset_size as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    let trailing_elements = interleaving_depth.saturating_mul(codeword_length - coset_size);
    if trailing_elements != 0 {
        encode_kernel(
            &command,
            &rt.replicate_first_coset,
            &[&current],
            &ReplicateCosetsParams {
                row_len: codeword_length as u32,
                coset_size: coset_size as u32,
                trailing_elements: trailing_elements as u32,
            },
            trailing_elements,
        );
    }

    encode_kernel(
        &command,
        &rt.apply_coset_twiddles,
        &[&current, &codeword_root_powers.limbs],
        &ApplyCosetTwiddlesParams {
            row_count: interleaving_depth as u32,
            num_cosets: num_cosets as u32,
            coset_size: coset_size as u32,
            codeword_length: codeword_length as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    let stage_count = coset_size.trailing_zeros() as usize;
    encode_kernel(
        &command,
        &rt.bit_reverse_rows,
        &[&current],
        &BitReverseParams {
            row_len: coset_size as u32,
            log_n: stage_count as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
        },
        total_elements,
    );

    let total_butterflies = total_elements / 2;
    let mut twiddle_offset = 0usize;
    for stage in 0..stage_count {
        let half_m = 1usize << stage;
        encode_kernel(
            &command,
            &rt.ntt_stage_rows,
            &[&current, &coset_roots.limbs],
            &NttStageParams {
                row_len: coset_size as u32,
                half_m: half_m as u32,
                twiddle_offset: twiddle_offset as u32,
                _pad0: 0,
            },
            total_butterflies,
        );
        twiddle_offset += 1usize << stage;
    }

    encode_kernel(
        &command,
        &rt.transpose,
        &[&current, &transposed],
        &TransposeParams {
            rows: interleaving_depth as u32,
            cols: codeword_length as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    wait_for_command_named(&command, "bn254_rs_encode");

    MetalBuffer::from_field_limb_buffer(transposed, total_elements)
}

pub(crate) fn encode_kernel<P: Copy>(
    command: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    params: &P,
    threads: usize,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        buffers.len() as u64,
        size_of::<P>() as u64,
        (params as *const P).cast::<c_void>(),
    );
    dispatch(&encoder, pipeline, threads.max(1));
    encoder.end_encoding();
}

pub(crate) fn roots_buffer(codeword_length: usize) -> MetalFieldBuffer {
    let rt = runtime();
    if let Some(buffer) = rt.ntt_roots.lock().unwrap().get(&codeword_length).cloned() {
        return buffer;
    }
    let root = Field256::get_root_of_unity(codeword_length as u64)
        .expect("BN254 root of unity unavailable for Metal NTT");
    let stage_count = codeword_length.trailing_zeros() as usize;
    let mut roots = Vec::with_capacity(codeword_length.saturating_sub(1));
    for stage in 0..stage_count {
        let stage_size = 1usize << (stage + 1);
        let half_stage = stage_size >> 1;
        let stage_root = root.pow([(codeword_length / stage_size) as u64]);
        let mut current = Field256::ONE;
        for _ in 0..half_stage {
            roots.push(current);
            current *= stage_root;
        }
    }
    let buffer = upload_field(&roots);
    rt.ntt_roots
        .lock()
        .unwrap()
        .insert(codeword_length, buffer.clone());
    buffer
}

pub(crate) fn root_powers_buffer(codeword_length: usize) -> MetalFieldBuffer {
    let rt = runtime();
    if let Some(buffer) = rt
        .root_powers
        .lock()
        .unwrap()
        .get(&codeword_length)
        .cloned()
    {
        return buffer;
    }
    let root = Field256::get_root_of_unity(codeword_length as u64)
        .expect("BN254 root of unity unavailable for Metal RS twiddles");
    let mut powers = Vec::with_capacity(codeword_length);
    let mut current = Field256::ONE;
    for _ in 0..codeword_length {
        powers.push(current);
        current *= root;
    }
    let buffer = upload_field(&powers);
    rt.root_powers
        .lock()
        .unwrap()
        .insert(codeword_length, buffer.clone());
    buffer
}

pub(crate) fn encode_field_rows_le(input: &Buffer, rows: usize, cols: usize) -> Buffer {
    assert!(rows <= u32::MAX as usize);
    assert!(cols <= u32::MAX as usize);
    let rt = runtime();
    let output = new_shared_buffer(rt, (rows * cols * size_of::<Field256>()) as u64);
    let command = rt.queue.new_command_buffer();
    encode_kernel(
        &command,
        &rt.encode_field_rows_le,
        &[input, &output],
        &FieldBytesParams {
            rows: rows as u32,
            cols: cols as u32,
        },
        rows * cols,
    );
    wait_for_command_named(&command, "bn254_encode_field_rows_le");
    output
}

pub(crate) fn read_bn254_rows(
    source: &MetalFieldBuffer,
    num_cols: usize,
    indices: &[usize],
) -> Vec<Field256> {
    if indices.is_empty() || num_cols == 0 {
        return Vec::new();
    }
    assert!(num_cols <= u32::MAX as usize);
    assert!(indices.iter().all(|&index| index <= u32::MAX as usize));
    let total = indices
        .len()
        .checked_mul(num_cols)
        .expect("Metal read_rows size overflow");
    assert!(total <= u32::MAX as usize);

    let rt = runtime();
    let indices = indices
        .iter()
        .copied()
        .map(|index| index as u32)
        .collect::<Vec<_>>();
    let indices_buffer = new_shared_buffer_with_data(
        rt,
        indices.as_ptr().cast(),
        (indices.len() * size_of::<u32>()) as u64,
    );
    let out = new_shared_buffer(rt, (total * size_of::<Field256>()) as u64);
    run_in_place(
        "bn254_read_rows",
        &[&source.limbs, &indices_buffer, &out],
        &[num_cols as u32, total as u32],
        total,
    );
    download_field(&out, total)
}

pub(crate) fn upload_field(values: &[Field256]) -> MetalFieldBuffer {
    let rt = runtime();
    let buffer = if values.is_empty() {
        new_shared_buffer(rt, 0)
    } else {
        // Field256 is 4 contiguous u64 limbs; upload directly without
        // flattening into an intermediate Vec.
        new_shared_buffer_with_data(
            rt,
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
        )
    };
    MetalFieldBuffer { limbs: buffer }
}

pub(crate) fn zeroed_field_buffer(len: usize) -> MetalFieldBuffer {
    let rt = runtime();
    let bytes = (len * size_of::<Field256>()) as u64;
    let buffer = new_shared_buffer(rt, bytes);
    if bytes > 0 {
        let command = rt.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(&buffer, metal::NSRange::new(0, bytes), 0);
        blit.end_encoding();
        wait_for_blit(command, bytes);
    }
    MetalFieldBuffer { limbs: buffer }
}

pub(crate) fn maybe_upload_bn254<T: Clone>(values: &[T]) -> Option<MetalFieldBuffer> {
    (type_name::<T>() == type_name::<Field256>()).then(|| upload_field(as_field256_slice(values)))
}

/// Geometric accumulate into `field[offset .. offset+len]` by binding the
/// field buffer at a byte offset; the kernel itself indexes from `gid == 0`.
pub(crate) fn geometric_accumulate_at_offset(
    field: &MetalFieldBuffer,
    offset: usize,
    len: usize,
    points: &MetalFieldBuffer,
    scalars: &MetalFieldBuffer,
    num_points: usize,
) {
    if len == 0 || num_points == 0 {
        return;
    }
    let rt = runtime();
    let command = rt.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    let pipe = pipeline(rt, "bn254_geometric_accumulate");
    encoder.set_compute_pipeline_state(pipe);
    let byte_offset = (offset * size_of::<Field256>()) as u64;
    encoder.set_buffer(0, Some(&field.limbs), byte_offset);
    encoder.set_buffer(1, Some(&points.limbs), 0);
    encoder.set_buffer(2, Some(&scalars.limbs), 0);
    let len_u32 = len as u32;
    let num_u32 = num_points as u32;
    encoder.set_bytes(3, size_of::<u32>() as u64, (&len_u32 as *const u32).cast());
    encoder.set_bytes(4, size_of::<u32>() as u64, (&num_u32 as *const u32).cast());
    dispatch(&encoder, pipe, len);
    encoder.end_encoding();
    wait_for_command_named(command, "bn254_geometric_accumulate_window");
}

/// In-place scalar multiply of `field[offset .. offset+len]` by binding the
/// field buffer at a byte offset; the kernel indexes from `gid == 0`.
pub(crate) fn scalar_mul_at_offset(
    field: &MetalFieldBuffer,
    offset: usize,
    len: usize,
    weight: &MetalFieldBuffer,
) {
    if len == 0 {
        return;
    }
    let rt = runtime();
    let command = rt.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    let pipe = pipeline(rt, "bn254_scalar_mul");
    encoder.set_compute_pipeline_state(pipe);
    let byte_offset = (offset * size_of::<Field256>()) as u64;
    encoder.set_buffer(0, Some(&field.limbs), byte_offset);
    encoder.set_buffer(1, Some(&weight.limbs), 0);
    let len_u32 = len as u32;
    encoder.set_bytes(2, size_of::<u32>() as u64, (&len_u32 as *const u32).cast());
    dispatch(&encoder, pipe, len);
    encoder.end_encoding();
    wait_for_command_named(command, "bn254_scalar_mul_window");
}

/// Multilinear extension of `field[off..off+len]` evaluated at `point`.
pub(crate) fn parallel_multilinear_extend_at(
    field: &MetalFieldBuffer,
    off: usize,
    len: usize,
    point: &MetalFieldBuffer,
    num_vars: usize,
) -> Field256 {
    let rt = runtime();
    let out = new_shared_buffer(rt, (4 * size_of::<u64>()) as u64);
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel_with_offsets(
        command,
        pipeline(rt, "bn254_multilinear_extend"),
        &[&field.limbs, &point.limbs, &out],
        &[field_byte_offset(off), 0, 0],
        &[len as u32, num_vars as u32],
        1,
    );
    wait_for_command_named(command, "bn254_multilinear_extend");
    download_field(&out, 1)[0]
}

/// `acc[acc_off..] += weight * vector[vec_off..]` over `len` elements.
pub(crate) fn scalar_mul_add_at(
    acc: &MetalFieldBuffer,
    acc_off: usize,
    vector: &MetalFieldBuffer,
    vec_off: usize,
    weight: &MetalFieldBuffer,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let rt = runtime();
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel_with_offsets(
        command,
        pipeline(rt, "bn254_scalar_mul_add"),
        &[&acc.limbs, &vector.limbs, &weight.limbs],
        &[field_byte_offset(acc_off), field_byte_offset(vec_off), 0],
        &[len as u32],
        len,
    );
    wait_for_command_named(command, "bn254_scalar_mul_add");
}

pub(crate) fn copy_field_buffer(source: &MetalFieldBuffer, len: usize) -> MetalFieldBuffer {
    copy_field_buffer_at(source, 0, len)
}

pub(crate) fn copy_field_buffer_at(
    source: &MetalFieldBuffer,
    offset: usize,
    len: usize,
) -> MetalFieldBuffer {
    let rt = runtime();
    let byte_len = (len * 4 * size_of::<u64>()) as u64;
    let source_offset = (offset * 4 * size_of::<u64>()) as u64;
    let target = new_shared_buffer(rt, byte_len);
    let command = rt.queue.new_command_buffer();
    let blit = command.new_blit_command_encoder();
    blit.copy_from_buffer(&source.limbs, source_offset, &target, 0, byte_len);
    blit.end_encoding();
    wait_for_blit(&command, byte_len);
    MetalFieldBuffer { limbs: target }
}

/// Encodes a kernel dispatch with `u32` constants bound as inline bytes
/// (no per-constant buffer allocations).
pub(crate) fn encode_u32_kernel(
    command: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    constants: &[u32],
    threads: usize,
) {
    encode_u32_kernel_with_offsets(command, pipeline, buffers, &[], constants, threads);
}

/// Like [`encode_u32_kernel`], but binds `buffers[i]` at byte offset
/// `offsets[i]` (defaulting to 0 when `offsets` is shorter). This is the
/// mechanism that lets a view dispatch a kernel over `parent[offset..]`
/// without copying: only the input binding shifts, the kernel still indexes
/// from `gid == 0`.
pub(crate) fn encode_u32_kernel_with_offsets(
    command: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    offsets: &[u64],
    constants: &[u32],
    threads: usize,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    let mut index = 0;
    for (i, buffer) in buffers.iter().enumerate() {
        let byte_offset = offsets.get(i).copied().unwrap_or(0);
        encoder.set_buffer(index, Some(buffer), byte_offset);
        index += 1;
    }
    for constant in constants {
        encoder.set_bytes(
            index,
            size_of::<u32>() as u64,
            (constant as *const u32).cast(),
        );
        index += 1;
    }
    dispatch(&encoder, pipeline, threads.max(1));
    encoder.end_encoding();
}

/// Byte offset of element `offset` within a `Field256` buffer.
#[inline]
pub(crate) fn field_byte_offset(offset: usize) -> u64 {
    (offset * size_of::<Field256>()) as u64
}

pub(crate) fn run_in_place(name: &str, buffers: &[&Buffer], constants: &[u32], threads: usize) {
    let rt = runtime();
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(command, pipeline(rt, name), buffers, constants, threads);
    wait_for_command_named(command, name);
}

pub(crate) fn dispatch(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: usize,
) {
    // Use full threadgroups (capped at 256) instead of a single execution
    // width; the pipeline limit already accounts for register pressure.
    let width = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let group_width = width.min(threads as u64).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: threads as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    );
}

pub(crate) fn download_field(buffer: &Buffer, len: usize) -> Vec<Field256> {
    if len == 0 {
        return Vec::new();
    }
    let start = Instant::now();
    let limbs = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u64>(), len * 4) };
    let result = limbs
        .chunks_exact(4)
        .map(|chunk| {
            Fp::<MontBackend<BN254Config, 4>, 4>(
                BigInt([chunk[0], chunk[1], chunk[2], chunk[3]]),
                PhantomData,
            )
        })
        .collect();
    profile::record_readback((len * size_of::<Field256>()) as u64, start.elapsed());
    result
}

pub(crate) fn download_field_at(buffer: &Buffer, index: usize) -> Field256 {
    let start = Instant::now();
    let limbs =
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u64>().add(index * 4), 4) };
    let result = Fp::<MontBackend<BN254Config, 4>, 4>(
        BigInt([limbs[0], limbs[1], limbs[2], limbs[3]]),
        PhantomData,
    );
    profile::record_readback(size_of::<Field256>() as u64, start.elapsed());
    result
}

pub(crate) fn download_hash_indices(buffer: &Buffer, len: usize, indices: &[usize]) -> Vec<Digest> {
    let start = Instant::now();
    let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), len * 32) };
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        assert!(index < len, "Metal hash index out of bounds");
        let mut hash = Digest::default();
        hash.as_mut_bytes()
            .copy_from_slice(&bytes[index * 32..(index + 1) * 32]);
        result.push(hash);
    }
    profile::record_readback(
        (indices.len() * size_of::<Digest>()) as u64,
        start.elapsed(),
    );
    result
}

pub(crate) fn assert_bn254<F>() {
    assert_eq!(
        type_name::<F>(),
        type_name::<Field256>(),
        "MetalBuffer only supports BN254 Field256 field operations"
    );
}

pub(crate) fn f_to_field256<F: Field>(value: F) -> Field256 {
    assert_bn254::<F>();
    unsafe { std::mem::transmute_copy(&value) }
}

pub(crate) fn field256_to_f<F: Field>(value: Field256) -> F {
    assert_bn254::<F>();
    unsafe { std::mem::transmute_copy(&value) }
}

pub(crate) fn target_to_field256<T: Field>(value: T) -> Field256 {
    assert_bn254::<T>();
    unsafe { std::mem::transmute_copy(&value) }
}

pub(crate) fn field256_to_target<T: Field>(value: Field256) -> T {
    assert_bn254::<T>();
    unsafe { std::mem::transmute_copy(&value) }
}

pub(crate) fn as_field256_slice<T: Clone>(values: &[T]) -> &[Field256] {
    assert_eq!(
        type_name::<T>(),
        type_name::<Field256>(),
        "MetalBuffer only supports BN254 Field256 buffers"
    );
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}
