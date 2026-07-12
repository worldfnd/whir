pub mod algebra;
pub mod ark_serde;
pub mod bits;
pub mod buffer;
pub mod cmdline_utils;
pub mod engines;
pub mod hash;
pub mod parameters;
pub mod protocols;
pub mod transcript;
pub mod type_info;
pub mod type_map;
pub mod utils; // Utils in general

#[cfg(test)]
mod tests {
    use std::sync::Once;

    pub fn init() {
        static INIT: Once = Once::new();

        #[cfg(not(feature = "tracing"))]
        INIT.call_once(|| {});

        #[cfg(feature = "tracing")]
        INIT.call_once(|| {
            use tracing_subscriber::{fmt, fmt::format::FmtSpan, EnvFilter};

            // Respect RUST_LOG if set, otherwise default for tests.
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

            // Create a writer compatible with the testing harnesses
            fmt()
                .with_env_filter(filter)
                .with_span_events(FmtSpan::ENTER)
                .with_test_writer()
                .init();

            tracing::debug!("Initialized test logger");
        });
    }
}
