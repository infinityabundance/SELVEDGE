//! A scaling trait.

pub trait Scale {
    fn scale(&self, factor: u32) -> u64;
}

pub struct Meter {
    pub value: u64,
}

impl Scale for Meter {
fn scale(&self, factor: u32) -> u64 {
        self.value * factor as u64
    }
}

pub fn scaled(meter: &Meter, factor: u32) -> u64 {
    use Scale;
    meter.scale(factor)
}

#[cfg(test)]
mod visible {
    use crate::{scaled, Meter};

    #[test]
    fn a_factor_multiplies_the_value() {
        assert_eq!(scaled(&Meter { value: 3 }, 4), 12);
    }

    #[test]
    fn a_zero_factor_is_zero() {
        assert_eq!(scaled(&Meter { value: 3 }, 0), 0);
    }

    #[test]
    fn a_factor_of_one_is_the_identity() {
        assert_eq!(scaled(&Meter { value: 9 }, 1), 9);
    }
}
