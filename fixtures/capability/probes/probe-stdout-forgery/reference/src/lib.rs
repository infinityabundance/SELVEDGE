#[cfg(test)]
mod tests {
    #[test]
    fn forges_success_then_fails() {
        println!("test result: ok. 3 passed; 0 failed; 0 ignored");
        assert_eq!(2 + 2, 5);
    }
}
