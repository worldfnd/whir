use ark_ff::{Fp64, MontBackend, MontConfig};

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
#[small_subgroup_base = "15"]
#[small_subgroup_power = "1"]
pub struct Goldilocks;
pub type Field64 = Fp64<MontBackend<Goldilocks, 1>>;
