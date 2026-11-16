//! Address normalisation: only the domain is lowercased.

pub fn normalise(text: &str) -> String {
    match text.split_once('@') {
        Some((local, domain)) => {
            format!("{}@{}", local, domain.to_lowercase())
        }
        None => text.to_string(),
    }
}

#[cfg(test)]
mod visible {
    use crate::normalise;

    #[test]
    fn the_local_part_keeps_its_case() {
        assert_eq!(normalise("User@Example.COM"), "User@example.com");
    }

    #[test]
    fn an_already_normal_address_is_unchanged() {
        assert_eq!(normalise("user@example.com"), "user@example.com");
    }

    #[test]
    fn text_without_a_domain_is_unchanged() {
        assert_eq!(normalise("NoAtSign"), "NoAtSign");
    }

    #[test]
    fn empty_input_is_empty() {
        assert_eq!(normalise(""), "");
    }
}
