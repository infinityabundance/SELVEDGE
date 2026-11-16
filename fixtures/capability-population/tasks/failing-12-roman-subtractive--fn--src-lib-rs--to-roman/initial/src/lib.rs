//! Roman numerals.

// TODO(FUNCTION)

#[cfg(test)]
mod visible {
    use crate::to_roman;

    #[test]
    fn every_subtractive_pair_is_used() {
        assert_eq!(to_roman(4), "IV");
        assert_eq!(to_roman(9), "IX");
        assert_eq!(to_roman(40), "XL");
        assert_eq!(to_roman(90), "XC");
        assert_eq!(to_roman(400), "CD");
        assert_eq!(to_roman(900), "CM");
    }

    #[test]
    fn a_compound_number_is_exact() {
        assert_eq!(to_roman(1994), "MCMXCIV");
    }

    #[test]
    fn the_bounds_are_respected() {
        assert_eq!(to_roman(1), "I");
        assert_eq!(to_roman(3999), "MMMCMXCIX");
        assert_eq!(to_roman(0), "");
        assert_eq!(to_roman(4000), "");
    }
}
