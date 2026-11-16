//! Roman numerals.

pub fn to_roman(value: u32) -> String {
    if value == 0 || value > 3999 {
        return String::new();
    }
    let table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ];
    let mut remaining = value;
    let mut out = String::new();
    for (amount, symbol) in table {
        while remaining >= amount {
            out.push_str(symbol);
            remaining -= amount;
        }
    }
    out
}

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
