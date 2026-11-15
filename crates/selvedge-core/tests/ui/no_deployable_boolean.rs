// There is no single deployable boolean. A physical-compliance verdict needs a
// typed evidence record, not `true`.
//@ expect: mismatched types
use selvedge_core::{PhysicalCompliance, ReadMeasurement};

fn main() {
    let _ = PhysicalCompliance::compliant(true);
    let _ = ReadMeasurement::Measured {
        audit: "receipt".to_owned(),
    };
}
