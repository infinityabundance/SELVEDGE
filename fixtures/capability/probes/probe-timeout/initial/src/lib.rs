#[cfg(test)]
mod tests {
    #[test]
    fn never_terminates() {
        loop {
            std::hint::spin_loop();
        }
    }
}
