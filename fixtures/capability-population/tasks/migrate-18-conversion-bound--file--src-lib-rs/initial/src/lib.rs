//! Unit scaling. This does not build: `u64` does not convert into
//! `f64` implicitly.

pub fn scale_units(value: u64) -> f64 {
    scale::scale(value)
}
