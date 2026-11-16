//! Luhn validation.

pub fn luhn_valid(text: &str) -> bool {
    let digits: Vec<u32> = match text
        .chars()
        .map(|character| character.to_digit(10))
        .collect::<Option<Vec<u32>>>()
    {
        Some(digits) => digits,
        None => return false,
    };
    if digits.len() < 2 {
        return false;
    }
    let mut sum = 0u32;
    for (offset, digit) in digits.iter().rev().enumerate() {
        if offset % 2 == 1 {
            let doubled = digit * 2;
            sum += if doubled > 9 { doubled - 9 } else { doubled };
        } else {
            sum += digit;
        }
    }
    sum % 10 == 0
}
