use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MetalProfileSnapshot {
    pub upload_count: u64,
    pub upload_bytes: u64,
    pub upload_nanos: u64,
    pub readback_count: u64,
    pub readback_bytes: u64,
    pub readback_nanos: u64,
    pub alloc_count: u64,
    pub alloc_bytes: u64,
    pub command_count: u64,
    pub command_wait_nanos: u64,
    pub blit_count: u64,
    pub blit_bytes: u64,
    pub blit_wait_nanos: u64,
}

impl MetalProfileSnapshot {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            upload_count: self.upload_count.saturating_sub(before.upload_count),
            upload_bytes: self.upload_bytes.saturating_sub(before.upload_bytes),
            upload_nanos: self.upload_nanos.saturating_sub(before.upload_nanos),
            readback_count: self.readback_count.saturating_sub(before.readback_count),
            readback_bytes: self.readback_bytes.saturating_sub(before.readback_bytes),
            readback_nanos: self.readback_nanos.saturating_sub(before.readback_nanos),
            alloc_count: self.alloc_count.saturating_sub(before.alloc_count),
            alloc_bytes: self.alloc_bytes.saturating_sub(before.alloc_bytes),
            command_count: self.command_count.saturating_sub(before.command_count),
            command_wait_nanos: self
                .command_wait_nanos
                .saturating_sub(before.command_wait_nanos),
            blit_count: self.blit_count.saturating_sub(before.blit_count),
            blit_bytes: self.blit_bytes.saturating_sub(before.blit_bytes),
            blit_wait_nanos: self.blit_wait_nanos.saturating_sub(before.blit_wait_nanos),
        }
    }

    pub fn upload_ms(self) -> f64 {
        nanos_to_ms(self.upload_nanos)
    }

    pub fn readback_ms(self) -> f64 {
        nanos_to_ms(self.readback_nanos)
    }

    pub fn command_wait_ms(self) -> f64 {
        nanos_to_ms(self.command_wait_nanos)
    }

    pub fn blit_wait_ms(self) -> f64 {
        nanos_to_ms(self.blit_wait_nanos)
    }
}

static UPLOAD_COUNT: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static UPLOAD_NANOS: AtomicU64 = AtomicU64::new(0);
static READBACK_COUNT: AtomicU64 = AtomicU64::new(0);
static READBACK_BYTES: AtomicU64 = AtomicU64::new(0);
static READBACK_NANOS: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static COMMAND_COUNT: AtomicU64 = AtomicU64::new(0);
static COMMAND_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
static BLIT_COUNT: AtomicU64 = AtomicU64::new(0);
static BLIT_BYTES: AtomicU64 = AtomicU64::new(0);
static BLIT_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);

pub fn snapshot() -> MetalProfileSnapshot {
    MetalProfileSnapshot {
        upload_count: UPLOAD_COUNT.load(Ordering::Relaxed),
        upload_bytes: UPLOAD_BYTES.load(Ordering::Relaxed),
        upload_nanos: UPLOAD_NANOS.load(Ordering::Relaxed),
        readback_count: READBACK_COUNT.load(Ordering::Relaxed),
        readback_bytes: READBACK_BYTES.load(Ordering::Relaxed),
        readback_nanos: READBACK_NANOS.load(Ordering::Relaxed),
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        command_count: COMMAND_COUNT.load(Ordering::Relaxed),
        command_wait_nanos: COMMAND_WAIT_NANOS.load(Ordering::Relaxed),
        blit_count: BLIT_COUNT.load(Ordering::Relaxed),
        blit_bytes: BLIT_BYTES.load(Ordering::Relaxed),
        blit_wait_nanos: BLIT_WAIT_NANOS.load(Ordering::Relaxed),
    }
}

pub fn record_upload(bytes: u64, duration: Duration) {
    UPLOAD_COUNT.fetch_add(1, Ordering::Relaxed);
    UPLOAD_BYTES.fetch_add(bytes, Ordering::Relaxed);
    UPLOAD_NANOS.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
}

pub fn record_readback(bytes: u64, duration: Duration) {
    READBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    READBACK_BYTES.fetch_add(bytes, Ordering::Relaxed);
    READBACK_NANOS.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
}

pub fn record_alloc(bytes: u64) {
    ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

pub fn record_command_wait(duration: Duration) {
    COMMAND_COUNT.fetch_add(1, Ordering::Relaxed);
    COMMAND_WAIT_NANOS.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
}

pub fn record_blit(bytes: u64, duration: Duration) {
    BLIT_COUNT.fetch_add(1, Ordering::Relaxed);
    BLIT_BYTES.fetch_add(bytes, Ordering::Relaxed);
    BLIT_WAIT_NANOS.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
}

fn nanos_to_ms(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}
