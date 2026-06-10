use std::{
    alloc::{GlobalAlloc, Layout, System},
    borrow::Cow,
    fs::OpenOptions,
    hint::black_box,
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use clap::Parser;
use serde::Serialize;
use whir::{
    algebra::{
        buffer::{ActiveBuffer, FieldOps},
        embedding::Identity,
        fields::Field256,
    },
    hash::{self, HASH_COUNTER},
    parameters::ProtocolParameters,
    protocols::whir::Config as WhirConfig,
    transcript::{codecs::Empty, DomainSeparator, ProverState},
};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator;

static CURRENT_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add_allocated(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            match new_size.cmp(&layout.size()) {
                std::cmp::Ordering::Greater => add_allocated(new_size - layout.size()),
                std::cmp::Ordering::Less => {
                    CURRENT_ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
                }
                std::cmp::Ordering::Equal => {}
            }
        }
        new_ptr
    }
}

fn add_allocated(bytes: usize) {
    let current = CURRENT_ALLOCATED.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_ALLOCATED.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_ALLOCATED.compare_exchange_weak(
            peak,
            current,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(next) => peak = next,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Phase benchmark for CPU/GPU WHIR proving")]
struct Args {
    #[arg(long, default_value_t = 16)]
    min_log_size: usize,

    #[arg(long, default_value_t = 28)]
    max_log_size: usize,

    #[arg(long, default_value = "1,2,3,4,6")]
    folds: String,

    #[arg(long, default_value = "1,2,3")]
    rates: String,

    /// Apply the article-style cap where rates above 1 are skipped for n >= 24.
    #[arg(long)]
    article_grid: bool,

    #[arg(long, default_value = "commit,sumcheck,e2e")]
    phases: String,

    #[arg(long, default_value_t = 128)]
    security_level: usize,

    #[arg(long, default_value_t = 20)]
    pow_bits: usize,

    #[arg(long, default_value_t = false)]
    unique_decoding: bool,

    #[arg(long, default_value = "outputs/phase_benchmark.jsonl")]
    output: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Commit,
    Sumcheck,
    E2e,
}

#[derive(Serialize)]
struct PhaseRow {
    backend: &'static str,
    phase: &'static str,
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    duration_ms: f64,
    current_allocated_bytes: usize,
    peak_allocated_bytes: usize,
    peak_phase_delta_bytes: usize,
    hashes: usize,
    proof_bytes: Option<usize>,
    status: &'static str,
    metal_upload_count: u64,
    metal_upload_bytes: u64,
    metal_upload_ms: f64,
    metal_readback_count: u64,
    metal_readback_bytes: u64,
    metal_readback_ms: f64,
    metal_alloc_count: u64,
    metal_alloc_bytes: u64,
    metal_command_count: u64,
    metal_command_wait_ms: f64,
    metal_blit_count: u64,
    metal_blit_bytes: u64,
    metal_blit_wait_ms: f64,
}

struct Measurement<T> {
    value: T,
    duration: Duration,
    current_allocated_bytes: usize,
    peak_allocated_bytes: usize,
    peak_phase_delta_bytes: usize,
    metal: MetalPhaseProfile,
}

#[derive(Clone, Copy, Debug, Default)]
struct MetalPhaseProfile {
    upload_count: u64,
    upload_bytes: u64,
    upload_ms: f64,
    readback_count: u64,
    readback_bytes: u64,
    readback_ms: f64,
    alloc_count: u64,
    alloc_bytes: u64,
    command_count: u64,
    command_wait_ms: f64,
    blit_count: u64,
    blit_bytes: u64,
    blit_wait_ms: f64,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
const BACKEND: &str = "gpu-metal";

#[cfg(not(all(feature = "metal", target_os = "macos")))]
const BACKEND: &str = "cpu";

type F = Field256;
type M = Identity<F>;

fn main() {
    let args = Args::parse();
    assert!(args.min_log_size <= args.max_log_size);
    warm_backend();

    let folds = parse_usize_list(&args.folds);
    let rates = parse_usize_list(&args.rates);
    let phases = parse_phases(&args.phases);

    std::fs::create_dir_all("outputs").unwrap();
    let mut output = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&args.output)
        .unwrap();

    for log_size in args.min_log_size..=args.max_log_size {
        for &fold in &folds {
            if fold > log_size {
                eprintln!("skip n={log_size} fold={fold}: fold must be <= n");
                continue;
            }
            for &rate in &rates {
                if args.article_grid && log_size >= 24 && rate > 1 {
                    continue;
                }
                let size = 1usize << log_size;
                let params = whir_params(&args, fold, rate);
                for &phase in &phases {
                    let row = match phase {
                        Phase::Commit => bench_commit(log_size, size, fold, rate, &params),
                        Phase::Sumcheck => bench_sumcheck(log_size, size, fold, rate, &params),
                        Phase::E2e => bench_e2e(log_size, size, fold, rate, &params),
                    };
                    writeln!(output, "{}", serde_json::to_string(&row).unwrap()).unwrap();
                    output.flush().unwrap();
                    eprintln!(
                        "{} n={log_size} fold={fold} rate={rate} phase={} {:.3} ms peak={} MiB upload={} KiB/{:.3} ms readback={} KiB/{:.3} ms cmd_wait={:.3} ms",
                        BACKEND,
                        row.phase,
                        row.duration_ms,
                        row.peak_allocated_bytes / (1024 * 1024),
                        row.metal_upload_bytes / 1024,
                        row.metal_upload_ms,
                        row.metal_readback_bytes / 1024,
                        row.metal_readback_ms,
                        row.metal_command_wait_ms,
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn warm_backend() {
    whir::algebra::buffer::MetalBuffer::<F>::warmup();
    whir::hash::MetalSha2::warmup();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn warm_backend() {}

fn whir_params(args: &Args, fold: usize, rate: usize) -> ProtocolParameters {
    ProtocolParameters {
        security_level: args.security_level,
        pow_bits: args.pow_bits,
        initial_folding_factor: fold,
        folding_factor: fold,
        unique_decoding: args.unique_decoding,
        starting_log_inv_rate: rate,
        batch_size: 1,
        hash_id: hash::SHA2,
    }
}

fn bench_commit(
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    whir_params: &ProtocolParameters,
) -> PhaseRow {
    let vector = input_vector(size);
    let vector_buffer = ActiveBuffer::from_slice(&vector);
    let params = WhirConfig::<M>::new(size, whir_params);
    let ds = DomainSeparator::protocol(&params)
        .session(&"phase benchmark commit")
        .instance(&Empty);

    HASH_COUNTER.reset();
    let measured = measure_phase(|| {
        let mut prover_state = ProverState::new_std(&ds);
        let _ = black_box(params.commit(&mut prover_state, &[&vector_buffer]));
    });
    row(
        Phase::Commit,
        log_size,
        size,
        fold,
        rate,
        measured,
        HASH_COUNTER.get(),
        None,
    )
}

fn bench_sumcheck(
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    whir_params: &ProtocolParameters,
) -> PhaseRow {
    let params = WhirConfig::<M>::new(size, whir_params);
    let config = params.initial_sumcheck.clone();
    let ds = DomainSeparator::protocol(&config)
        .session(&"phase benchmark sumcheck")
        .instance(&Empty);

    HASH_COUNTER.reset();
    let measured = measure_phase(|| {
        let mut a = ActiveBuffer::from_vec(input_vector(size));
        let mut b = ActiveBuffer::from_vec((0..size).map(|i| F::from(i as u64 + 17)).collect());
        let mut sum = a.dot(&b);
        let mut prover_state = ProverState::new_std(&ds);
        let _ = black_box(config.prove(&mut prover_state, &mut a, &mut b, &mut sum, &[]));
    });
    row(
        Phase::Sumcheck,
        log_size,
        size,
        fold,
        rate,
        measured,
        HASH_COUNTER.get(),
        None,
    )
}

fn bench_e2e(
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    whir_params: &ProtocolParameters,
) -> PhaseRow {
    let vector = input_vector(size);
    let vector_buffer = ActiveBuffer::from_slice(&vector);
    let params = WhirConfig::<M>::new(size, whir_params);
    let ds = DomainSeparator::protocol(&params)
        .session(&"phase benchmark e2e")
        .instance(&Empty);

    HASH_COUNTER.reset();
    let measured = measure_phase(|| {
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &[&vector_buffer]);
        let _ = params.prove(
            &mut prover_state,
            &[&vector_buffer],
            vec![&witness],
            vec![],
            Cow::Owned(vec![]),
        );
        let proof = prover_state.proof();
        proof.narg_string.len() + proof.hints.len()
    });
    let proof_bytes = measured.value;
    row(
        Phase::E2e,
        log_size,
        size,
        fold,
        rate,
        measured,
        HASH_COUNTER.get(),
        Some(proof_bytes),
    )
}

fn input_vector(size: usize) -> Vec<F> {
    (0..size).map(|i| F::from(i as u64)).collect()
}

fn measure_phase<T>(f: impl FnOnce() -> T) -> Measurement<T> {
    let before = CURRENT_ALLOCATED.load(Ordering::Relaxed);
    PEAK_ALLOCATED.store(before, Ordering::Relaxed);
    let metal_before = metal_profile_snapshot();
    let start = Instant::now();
    let value = f();
    let duration = start.elapsed();
    let metal_after = metal_profile_snapshot();
    let current = CURRENT_ALLOCATED.load(Ordering::Relaxed);
    let peak = PEAK_ALLOCATED.load(Ordering::Relaxed);
    Measurement {
        value,
        duration,
        current_allocated_bytes: current,
        peak_allocated_bytes: peak,
        peak_phase_delta_bytes: peak.saturating_sub(before),
        metal: metal_profile_delta(metal_before, metal_after),
    }
}

fn row<T>(
    phase: Phase,
    log_size: usize,
    size: usize,
    fold: usize,
    rate: usize,
    measurement: Measurement<T>,
    hashes: usize,
    proof_bytes: Option<usize>,
) -> PhaseRow {
    PhaseRow {
        backend: BACKEND,
        phase: phase.name(),
        log_size,
        size,
        fold,
        rate,
        duration_ms: measurement.duration.as_secs_f64() * 1_000.0,
        current_allocated_bytes: measurement.current_allocated_bytes,
        peak_allocated_bytes: measurement.peak_allocated_bytes,
        peak_phase_delta_bytes: measurement.peak_phase_delta_bytes,
        hashes,
        proof_bytes,
        status: "ok",
        metal_upload_count: measurement.metal.upload_count,
        metal_upload_bytes: measurement.metal.upload_bytes,
        metal_upload_ms: measurement.metal.upload_ms,
        metal_readback_count: measurement.metal.readback_count,
        metal_readback_bytes: measurement.metal.readback_bytes,
        metal_readback_ms: measurement.metal.readback_ms,
        metal_alloc_count: measurement.metal.alloc_count,
        metal_alloc_bytes: measurement.metal.alloc_bytes,
        metal_command_count: measurement.metal.command_count,
        metal_command_wait_ms: measurement.metal.command_wait_ms,
        metal_blit_count: measurement.metal.blit_count,
        metal_blit_bytes: measurement.metal.blit_bytes,
        metal_blit_wait_ms: measurement.metal.blit_wait_ms,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_profile_snapshot() -> whir::hash::MetalProfileSnapshot {
    whir::hash::metal_profile_snapshot()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_profile_snapshot() -> MetalProfileSnapshotCompat {
    MetalProfileSnapshotCompat::default()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[derive(Clone, Copy, Debug, Default)]
struct MetalProfileSnapshotCompat;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_profile_delta(
    before: whir::hash::MetalProfileSnapshot,
    after: whir::hash::MetalProfileSnapshot,
) -> MetalPhaseProfile {
    let delta = after.delta_since(before);
    MetalPhaseProfile {
        upload_count: delta.upload_count,
        upload_bytes: delta.upload_bytes,
        upload_ms: delta.upload_ms(),
        readback_count: delta.readback_count,
        readback_bytes: delta.readback_bytes,
        readback_ms: delta.readback_ms(),
        alloc_count: delta.alloc_count,
        alloc_bytes: delta.alloc_bytes,
        command_count: delta.command_count,
        command_wait_ms: delta.command_wait_ms(),
        blit_count: delta.blit_count,
        blit_bytes: delta.blit_bytes,
        blit_wait_ms: delta.blit_wait_ms(),
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_profile_delta(
    _before: MetalProfileSnapshotCompat,
    _after: MetalProfileSnapshotCompat,
) -> MetalPhaseProfile {
    MetalPhaseProfile::default()
}

impl Phase {
    const fn name(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::Sumcheck => "sumcheck",
            Self::E2e => "e2e_prove",
        }
    }
}

fn parse_usize_list(source: &str) -> Vec<usize> {
    source
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().unwrap())
        .collect()
}

fn parse_phases(source: &str) -> Vec<Phase> {
    source
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| match part {
            "commit" => Phase::Commit,
            "sumcheck" => Phase::Sumcheck,
            "e2e" | "e2e_prove" => Phase::E2e,
            other => panic!("unknown phase {other}; use commit,sumcheck,e2e"),
        })
        .collect()
}
