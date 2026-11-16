//! Luhn validation. The visible test fails: the doubling starts from
//! the wrong end.

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
    for (index, digit) in digits.iter().enumerate() {
        if index % 2 == 0 {
            let doubled = digit * 2;
            sum += if doubled > 9 { doubled - 9 } else { doubled };
        } else {
            sum += digit;
        }
    }
    sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use super::luhn_valid;

    #[test]
    fn visible_known_valid_number() {
        assert!(luhn_valid("49927398716"));
    }
}
