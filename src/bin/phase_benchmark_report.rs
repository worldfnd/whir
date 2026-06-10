use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct Row {
    backend: String,
    phase: String,
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    duration_ms: f64,
    peak_allocated_bytes: usize,
    peak_phase_delta_bytes: usize,
    hashes: usize,
    proof_bytes: Option<usize>,
    status: Option<String>,
    #[serde(default)]
    metal_upload_bytes: u64,
    #[serde(default)]
    metal_upload_ms: f64,
    #[serde(default)]
    metal_readback_bytes: u64,
    #[serde(default)]
    metal_readback_ms: f64,
    #[serde(default)]
    metal_alloc_bytes: u64,
    #[serde(default)]
    metal_command_wait_ms: f64,
    #[serde(default)]
    metal_blit_bytes: u64,
    #[serde(default)]
    metal_blit_wait_ms: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    phase: String,
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
}

#[derive(Default)]
struct Pair {
    cpu: Option<Row>,
    gpu: Option<Row>,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "outputs/phase_benchmark_cpu_gpu.jsonl".to_string());
    let file = File::open(&path).unwrap_or_else(|err| panic!("failed to open {path}: {err}"));
    let mut rows = BTreeMap::<Key, Pair>::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(&line).unwrap();
        let key = Key {
            phase: row.phase.clone(),
            log_size: row.log_size,
            size: row.size,
            fold: row.fold,
            rate: row.rate,
        };
        let pair = rows.entry(key).or_default();
        match row.backend.as_str() {
            "cpu" => pair.cpu = Some(row),
            "gpu-metal" => pair.gpu = Some(row),
            _ => {}
        }
    }

    println!(
        "log_size,size,fold,rate,phase,cpu_status,gpu_status,cpu_ms,gpu_ms,speedup,cpu_peak_bytes,gpu_peak_bytes,cpu_peak_delta_bytes,gpu_peak_delta_bytes,cpu_hashes,gpu_hashes,proof_bytes,gpu_upload_bytes,gpu_upload_ms,gpu_readback_bytes,gpu_readback_ms,gpu_alloc_bytes,gpu_command_wait_ms,gpu_blit_bytes,gpu_blit_wait_ms"
    );
    for (key, pair) in rows {
        let (Some(cpu), Some(gpu)) = (pair.cpu, pair.gpu) else {
            continue;
        };
        let speedup = if gpu.duration_ms > 0.0 {
            cpu.duration_ms / gpu.duration_ms
        } else {
            0.0
        };
        let proof_bytes = cpu.proof_bytes.or(gpu.proof_bytes).unwrap_or(0);
        let cpu_status = cpu.status.as_deref().unwrap_or("ok");
        let gpu_status = gpu.status.as_deref().unwrap_or("ok");
        println!(
            "{},{},{},{},{},{},{},{:.6},{:.6},{:.4},{},{},{},{},{},{},{},{},{:.6},{},{:.6},{},{:.6},{},{:.6}",
            key.log_size,
            key.size,
            key.fold,
            key.rate,
            key.phase,
            cpu_status,
            gpu_status,
            cpu.duration_ms,
            gpu.duration_ms,
            speedup,
            cpu.peak_allocated_bytes,
            gpu.peak_allocated_bytes,
            cpu.peak_phase_delta_bytes,
            gpu.peak_phase_delta_bytes,
            cpu.hashes,
            gpu.hashes,
            proof_bytes,
            gpu.metal_upload_bytes,
            gpu.metal_upload_ms,
            gpu.metal_readback_bytes,
            gpu.metal_readback_ms,
            gpu.metal_alloc_bytes,
            gpu.metal_command_wait_ms,
            gpu.metal_blit_bytes,
            gpu.metal_blit_wait_ms,
        );
    }
}
