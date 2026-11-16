//! CSV field quoting.

pub fn field(value: &str) -> String {
    
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }

}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn plain_fields_are_unquoted() {
        assert_eq!(field("plain"), "plain");
    }
}
