use std::{borrow::Cow, sync::OnceLock, time::Instant};

use const_oid::ObjectIdentifier;
use digest::const_oid::AssociatedOid;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use sha2::Sha256;
use zerocopy::IntoBytes;

use super::{metal_profile, Hash, HashEngine, HASH_COUNTER};

const USE_SHA256_64_SPECIALIZATION: bool = false;
const POW_MAX_WINDOW: u32 = 1 << 20;
const POW_MIN_WINDOW: u32 = 1 << 12;
const POW_THREADS_PER_GROUP: u64 = 256;

const SHA256_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint rotr(uint x, uint n) {
    return (x >> n) | (x << (32 - n));
}

inline uchar padded_byte(
    device const uchar *input,
    uint message,
    uint message_size,
    uint block_offset,
    uint padded_size
) {
    ulong bit_len = ((ulong)message_size) * 8UL;
    if (block_offset < message_size) {
        return input[message * message_size + block_offset];
    }
    if (block_offset == message_size) {
        return 0x80;
    }
    uint len_start = padded_size - 8;
    if (block_offset >= len_start) {
        uint shift = (7 - (block_offset - len_start)) * 8;
        return (uchar)((bit_len >> shift) & 0xff);
    }
    return 0;
}

kernel void sha256_many(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &message_size [[buffer(2)]],
    constant uint &messages [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= messages) return;

    uint padded_size = ((message_size + 9 + 63) / 64) * 64;
    uint h0 = 0x6a09e667;
    uint h1 = 0xbb67ae85;
    uint h2 = 0x3c6ef372;
    uint h3 = 0xa54ff53a;
    uint h4 = 0x510e527f;
    uint h5 = 0x9b05688c;
    uint h6 = 0x1f83d9ab;
    uint h7 = 0x5be0cd19;

    for (uint block = 0; block < padded_size; block += 64) {
        uint w[64];
        for (uint i = 0; i < 16; i++) {
            uint off = block + i * 4;
            w[i] =
                ((uint)padded_byte(input, gid, message_size, off + 0, padded_size) << 24) |
                ((uint)padded_byte(input, gid, message_size, off + 1, padded_size) << 16) |
                ((uint)padded_byte(input, gid, message_size, off + 2, padded_size) << 8) |
                ((uint)padded_byte(input, gid, message_size, off + 3, padded_size));
        }
        for (uint i = 16; i < 64; i++) {
            uint s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint a = h0, b = h1, c = h2, d = h3;
        uint e = h4, f = h5, g = h6, h = h7;
        for (uint i = 0; i < 64; i++) {
            uint s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint ch = (e & f) ^ ((~e) & g);
            uint temp1 = h + s1 + ch + K[i] + w[i];
            uint s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint maj = (a & b) ^ (a & c) ^ (b & c);
            uint temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    uint hs[8] = { h0, h1, h2, h3, h4, h5, h6, h7 };
    uint out = gid * 32;
    for (uint i = 0; i < 8; i++) {
        output[out + i * 4 + 0] = (uchar)((hs[i] >> 24) & 0xff);
        output[out + i * 4 + 1] = (uchar)((hs[i] >> 16) & 0xff);
        output[out + i * 4 + 2] = (uchar)((hs[i] >> 8) & 0xff);
        output[out + i * 4 + 3] = (uchar)(hs[i] & 0xff);
    }
}

inline uint load_be32(device const uchar *input, uint off) {
    return ((uint)input[off + 0] << 24) |
           ((uint)input[off + 1] << 16) |
           ((uint)input[off + 2] << 8) |
           ((uint)input[off + 3]);
}

inline void sha256_compress_16(thread uint h[8], thread uint w[16]) {
    uint a = h[0], b = h[1], c = h[2], d = h[3];
    uint e = h[4], f = h[5], g = h[6], hh = h[7];

    for (uint i = 0; i < 64; i++) {
        uint wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint s0 = rotr(w[(i + 1) & 15], 7) ^ rotr(w[(i + 1) & 15], 18) ^ (w[(i + 1) & 15] >> 3);
            uint s1 = rotr(w[(i + 14) & 15], 17) ^ rotr(w[(i + 14) & 15], 19) ^ (w[(i + 14) & 15] >> 10);
            wi = w[i & 15] + s0 + w[(i + 9) & 15] + s1;
            w[i & 15] = wi;
        }
        uint S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint temp1 = hh + S1 + ch + K[i] + wi;
        uint S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

kernel void sha256_many_64(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant uint &messages [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= messages) return;

    uint h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint w[16];
    uint base = gid * 64;
    for (uint i = 0; i < 16; i++) {
        w[i] = load_be32(input, base + i * 4);
    }
    sha256_compress_16(h, w);

    w[0] = 0x80000000;
    for (uint i = 1; i < 15; i++) {
        w[i] = 0;
    }
    w[15] = 512;
    sha256_compress_16(h, w);

    uint out = gid * 32;
    for (uint i = 0; i < 8; i++) {
        output[out + i * 4 + 0] = (uchar)((h[i] >> 24) & 0xff);
        output[out + i * 4 + 1] = (uchar)((h[i] >> 16) & 0xff);
        output[out + i * 4 + 2] = (uchar)((h[i] >> 8) & 0xff);
        output[out + i * 4 + 3] = (uchar)(h[i] & 0xff);
    }
}

inline ulong digest_threshold_value(uint h0, uint h1) {
    ulong b0 = (ulong)((h0 >> 24) & 0xff);
    ulong b1 = (ulong)((h0 >> 16) & 0xff);
    ulong b2 = (ulong)((h0 >> 8) & 0xff);
    ulong b3 = (ulong)(h0 & 0xff);
    ulong b4 = (ulong)((h1 >> 24) & 0xff);
    ulong b5 = (ulong)((h1 >> 16) & 0xff);
    ulong b6 = (ulong)((h1 >> 8) & 0xff);
    ulong b7 = (ulong)(h1 & 0xff);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) |
           (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56);
}

kernel void sha256_pow_64(
    device const uchar *challenge [[buffer(0)]],
    device ulong *candidates [[buffer(1)]],
    constant ulong &start_nonce [[buffer(2)]],
    constant ulong &threshold [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    threadgroup ulong local_best[256];
    ulong candidate = 0xffffffffffffffffUL;

    if (gid < count) {
        ulong nonce = start_nonce + (ulong)gid;

        uint h[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        uint w[16];
        for (uint i = 0; i < 8; i++) {
            w[i] = load_be32(challenge, i * 4);
        }
        w[8] = ((uint)(nonce & 0xffUL) << 24) |
               ((uint)((nonce >> 8) & 0xffUL) << 16) |
               ((uint)((nonce >> 16) & 0xffUL) << 8) |
               ((uint)((nonce >> 24) & 0xffUL));
        w[9] = ((uint)((nonce >> 32) & 0xffUL) << 24) |
               ((uint)((nonce >> 40) & 0xffUL) << 16) |
               ((uint)((nonce >> 48) & 0xffUL) << 8) |
               ((uint)((nonce >> 56) & 0xffUL));
        for (uint i = 10; i < 16; i++) {
            w[i] = 0;
        }
        sha256_compress_16(h, w);

        w[0] = 0x80000000;
        for (uint i = 1; i < 15; i++) {
            w[i] = 0;
        }
        w[15] = 512;
        sha256_compress_16(h, w);

        ulong value = digest_threshold_value(h[0], h[1]);
        if (value <= threshold) {
            candidate = nonce;
        }
    }

    local_best[tid] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && local_best[tid + stride] < local_best[tid]) {
            local_best[tid] = local_best[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        candidates[tgid] = local_best[0];
    }
}
"#;

#[derive(Clone, Copy, Debug, Default)]
pub struct MetalSha2;

struct MetalSha2Runtime {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pipeline_64: ComputePipelineState,
    pow_pipeline: ComputePipelineState,
}

fn new_shared_buffer(rt: &MetalSha2Runtime, bytes: u64) -> Buffer {
    metal_profile::record_alloc(bytes);
    rt.device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
}

fn new_shared_buffer_with_data(
    rt: &MetalSha2Runtime,
    data: *const std::ffi::c_void,
    bytes: u64,
) -> Buffer {
    let start = Instant::now();
    let buffer = rt
        .device
        .new_buffer_with_data(data, bytes, MTLResourceOptions::StorageModeShared);
    metal_profile::record_alloc(bytes);
    metal_profile::record_upload(bytes, start.elapsed());
    buffer
}

fn upload_u32(rt: &MetalSha2Runtime, value: u32) -> Buffer {
    new_shared_buffer_with_data(rt, (&value as *const u32).cast(), size_of::<u32>() as u64)
}

fn upload_u64(rt: &MetalSha2Runtime, value: u64) -> Buffer {
    new_shared_buffer_with_data(rt, (&value as *const u64).cast(), size_of::<u64>() as u64)
}

fn wait_for_command(command: &metal::CommandBufferRef) {
    wait_for_command_named(command, "sha256");
}

fn wait_for_command_named(command: &metal::CommandBufferRef, label: &str) {
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
    metal_profile::record_command_wait(elapsed);
}

fn runtime() -> &'static MetalSha2Runtime {
    static RUNTIME: OnceLock<MetalSha2Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let device = Device::system_default().expect("Metal device is not available");
        let library = device
            .new_library_with_source(SHA256_METAL, &CompileOptions::new())
            .expect("failed to compile Metal SHA-256 kernel");
        let function = library
            .get_function("sha256_many", None)
            .expect("missing Metal SHA-256 kernel");
        let function_64 = library
            .get_function("sha256_many_64", None)
            .expect("missing Metal SHA-256 64-byte kernel");
        let pow_function = library
            .get_function("sha256_pow_64", None)
            .expect("missing Metal SHA-256 PoW kernel");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("failed to create Metal SHA-256 pipeline");
        let pipeline_64 = device
            .new_compute_pipeline_state_with_function(&function_64)
            .expect("failed to create Metal SHA-256 64-byte pipeline");
        let pow_pipeline = device
            .new_compute_pipeline_state_with_function(&pow_function)
            .expect("failed to create Metal SHA-256 PoW pipeline");
        let queue = device.new_command_queue();
        MetalSha2Runtime {
            device,
            queue,
            pipeline,
            pipeline_64,
            pow_pipeline,
        }
    })
}

impl MetalSha2 {
    pub const fn new() -> Self {
        Self
    }

    pub fn warmup() {
        let _ = runtime();
    }

    pub fn prove_pow_64(challenge: &[u8; 32], threshold: u64) -> u64 {
        let rt = runtime();
        let window = pow_window_for_threshold(threshold);
        let challenge_buffer =
            new_shared_buffer_with_data(rt, challenge.as_ptr().cast(), challenge.len() as u64);
        let candidate_groups = u64::from(window).div_ceil(POW_THREADS_PER_GROUP);
        let candidate_bytes = candidate_groups * size_of::<u64>() as u64;
        let candidates = new_shared_buffer(rt, candidate_bytes);
        let threshold = upload_u64(rt, threshold);
        let count = upload_u32(rt, window);

        let mut start_nonce = 0u64;
        loop {
            let start = upload_u64(rt, start_nonce);
            let command = rt.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&rt.pow_pipeline);
            encoder.set_buffer(0, Some(&challenge_buffer), 0);
            encoder.set_buffer(1, Some(&candidates), 0);
            encoder.set_buffer(2, Some(&start), 0);
            encoder.set_buffer(3, Some(&threshold), 0);
            encoder.set_buffer(4, Some(&count), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: candidate_groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: POW_THREADS_PER_GROUP,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.end_encoding();
            wait_for_command_named(&command, "sha256_pow");

            let read_start = Instant::now();
            let candidates = unsafe {
                std::slice::from_raw_parts(
                    candidates.contents().cast::<u64>(),
                    candidate_groups as usize,
                )
            };
            let nonce = candidates.iter().copied().min().unwrap_or(u64::MAX);
            metal_profile::record_readback(candidate_bytes, read_start.elapsed());
            if nonce != u64::MAX {
                HASH_COUNTER.add((nonce - start_nonce + 1) as usize);
                return nonce;
            }
            HASH_COUNTER.add(window as usize);
            start_nonce = start_nonce
                .checked_add(window as u64)
                .expect("PoW nonce range exhausted");
        }
    }

    pub(crate) fn build_merkle_tree_buffer_from_messages_buffer(
        &self,
        message_size: usize,
        messages: &Buffer,
        num_leaves: usize,
        layers: usize,
    ) -> Buffer {
        assert_eq!(num_leaves, 1usize << layers);
        let rt = runtime();
        let num_nodes = (1usize << (layers + 1)) - 1;
        let nodes_bytes = (num_nodes * size_of::<Hash>()) as u64;
        let nodes = new_shared_buffer(rt, nodes_bytes);

        let command = rt.queue.new_command_buffer();
        let mut constants = Vec::with_capacity((layers + 1) * 2);
        self.encode_hash_many_into_command(
            &command,
            &mut constants,
            message_size,
            messages,
            0,
            num_leaves,
            &nodes,
            0,
        );

        let mut previous_offset = 0usize;
        let mut previous_len = num_leaves;
        let mut next_offset = num_leaves;
        for _ in 0..layers {
            let current_len = previous_len / 2;
            self.encode_hash_many_into_command(
                &command,
                &mut constants,
                64,
                &nodes,
                (previous_offset * size_of::<Hash>()) as u64,
                current_len,
                &nodes,
                (next_offset * size_of::<Hash>()) as u64,
            );
            previous_offset = next_offset;
            previous_len = current_len;
            next_offset += current_len;
        }
        wait_for_command_named(&command, "sha256_merkle_fused");
        drop(constants);
        nodes
    }

    fn hash_many_buffer_into_buffer(
        &self,
        size: usize,
        input_buffer: &Buffer,
        input_offset: u64,
        output_len: usize,
        output_buffer: &Buffer,
        output_offset: u64,
    ) {
        if output_len == 0 {
            return;
        }

        let command = runtime().queue.new_command_buffer();
        let mut constants = Vec::with_capacity(2);
        self.encode_hash_many_into_command(
            &command,
            &mut constants,
            size,
            input_buffer,
            input_offset,
            output_len,
            output_buffer,
            output_offset,
        );
        wait_for_command(&command);
        drop(constants);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_hash_many_into_command(
        &self,
        command: &metal::CommandBufferRef,
        constants: &mut Vec<Buffer>,
        size: usize,
        input_buffer: &Buffer,
        input_offset: u64,
        output_len: usize,
        output_buffer: &Buffer,
        output_offset: u64,
    ) {
        if output_len == 0 {
            return;
        }
        let rt = runtime();
        let messages_buffer = upload_u32(rt, output_len as u32);
        let encoder = command.new_compute_command_encoder();
        let pipeline = if size == 64 && USE_SHA256_64_SPECIALIZATION {
            encoder.set_compute_pipeline_state(&rt.pipeline_64);
            encoder.set_buffer(0, Some(input_buffer), input_offset);
            encoder.set_buffer(1, Some(output_buffer), output_offset);
            encoder.set_buffer(2, Some(&messages_buffer), 0);
            &rt.pipeline_64
        } else {
            let size_buffer = upload_u32(rt, size as u32);
            encoder.set_compute_pipeline_state(&rt.pipeline);
            encoder.set_buffer(0, Some(input_buffer), input_offset);
            encoder.set_buffer(1, Some(output_buffer), output_offset);
            encoder.set_buffer(2, Some(&size_buffer), 0);
            encoder.set_buffer(3, Some(&messages_buffer), 0);
            constants.push(size_buffer);
            &rt.pipeline
        };
        constants.push(messages_buffer);
        let group_width = pipeline
            .thread_execution_width()
            .min(output_len as u64)
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: output_len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        HASH_COUNTER.add(output_len);
    }

    pub(crate) fn hash_many_buffer(
        &self,
        size: usize,
        input_buffer: &Buffer,
        output_len: usize,
        output: &mut [Hash],
    ) {
        assert_eq!(output_len, output.len());
        if output.is_empty() {
            return;
        }

        let rt = runtime();

        let output_bytes = (output.len() * size_of::<Hash>()) as u64;
        let output_buffer = new_shared_buffer(rt, output_bytes);
        self.hash_many_buffer_into_buffer(size, input_buffer, 0, output.len(), &output_buffer, 0);

        let start = Instant::now();
        let bytes = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<u8>(), output.len() * 32)
        };
        for (out, bytes) in output.iter_mut().zip(bytes.chunks_exact(32)) {
            out.as_mut_bytes().copy_from_slice(bytes);
        }
        metal_profile::record_readback(output_bytes, start.elapsed());
    }
}

fn pow_window_for_threshold(threshold: u64) -> u32 {
    if threshold == 0 {
        return POW_MAX_WINDOW;
    }

    let expected_attempts = u64::MAX
        .checked_div(threshold)
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    let target = expected_attempts.saturating_mul(4);
    let window = target
        .checked_next_power_of_two()
        .unwrap_or(u64::from(POW_MAX_WINDOW))
        .clamp(u64::from(POW_MIN_WINDOW), u64::from(POW_MAX_WINDOW));
    window as u32
}

impl HashEngine for MetalSha2 {
    fn name(&self) -> Cow<'_, str> {
        "sha2".into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        Some(Sha256::OID)
    }

    fn supports_size(&self, _size: usize) -> bool {
        Device::system_default().is_some()
    }

    fn preferred_batch_size(&self) -> usize {
        256
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert_eq!(
            input.len(),
            size * output.len(),
            "Input length ({}) should be size * output.len() = {size} * {}",
            input.len(),
            output.len()
        );
        if output.is_empty() {
            return;
        }

        let input_buffer = if input.is_empty() {
            new_shared_buffer(runtime(), 0)
        } else {
            new_shared_buffer_with_data(runtime(), input.as_ptr().cast(), input.len() as u64)
        };
        self.hash_many_buffer(size, &input_buffer, output.len(), output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Sha2;

    #[test]
    fn metal_sha2_matches_cpu() {
        let rows = 17;
        let size = 73;
        let input = (0..rows * size)
            .map(|i: usize| (i.wrapping_mul(31) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut cpu = vec![Hash::default(); rows];
        let mut gpu = vec![Hash::default(); rows];
        Sha2::new().hash_many(size, &input, &mut cpu);
        MetalSha2::new().hash_many(size, &input, &mut gpu);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn metal_sha2_64_matches_cpu() {
        let rows = 257;
        let size = 64;
        let input = (0..rows * size)
            .map(|i: usize| (i.wrapping_mul(17).wrapping_add(3) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut cpu = vec![Hash::default(); rows];
        let mut gpu = vec![Hash::default(); rows];
        Sha2::new().hash_many(size, &input, &mut cpu);
        MetalSha2::new().hash_many(size, &input, &mut gpu);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn metal_sha2_pow_matches_threshold() {
        let challenge = [7u8; 32];
        let threshold = u64::MAX >> 8;
        let nonce = MetalSha2::prove_pow_64(&challenge, threshold);
        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&challenge);
        input[32..40].copy_from_slice(&nonce.to_le_bytes());
        let mut hash = Hash::default();
        Sha2::new().hash_many(64, &input, std::slice::from_mut(&mut hash));
        let value = u64::from_le_bytes(hash.0[..8].try_into().unwrap());
        assert!(value <= threshold);
    }
}
