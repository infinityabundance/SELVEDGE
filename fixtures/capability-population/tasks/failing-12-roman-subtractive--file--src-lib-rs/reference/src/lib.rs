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
