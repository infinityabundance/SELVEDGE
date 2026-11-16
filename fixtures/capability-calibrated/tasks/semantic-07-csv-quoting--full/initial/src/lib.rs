//! CSV field quoting, currently only for commas.

pub fn field(value: &str) -> String {
    if value.contains(',') {
        format!("\"{value}\"")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::field;

    #[test]
    fn quotes_a_comma_field() {
        assert_eq!(field("a,b"), "\"a,b\"");
    }
}
