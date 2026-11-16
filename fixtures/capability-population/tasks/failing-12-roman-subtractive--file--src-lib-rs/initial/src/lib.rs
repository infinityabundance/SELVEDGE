//! Roman numerals. The visible test fails: there are no subtractive
//! pairs.

pub fn to_roman(value: u32) -> String {
    if value == 0 || value > 3999 {
        return String::new();
    }
    let table = [
        (1000, "M"), (500, "D"), (100, "C"), (50, "L"),
        (10, "X"), (5, "V"), (1, "I"),
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
mod tests {
    use super::to_roman;

    #[test]
    fn visible_subtractive_four() {
        assert_eq!(to_roman(4), "IV");
    }
}
