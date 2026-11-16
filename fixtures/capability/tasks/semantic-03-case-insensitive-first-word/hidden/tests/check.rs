use semantic_03_case_insensitive_first_word::is_command;

#[test]
fn matches_regardless_of_case() {
    assert!(is_command("RUN now", "run"));
    assert!(is_command("Run now", "RUN"));
}

#[test]
fn only_the_first_word_counts() {
    assert!(!is_command("please run", "run"));
}

#[test]
fn a_prefix_is_not_a_word_match() {
    assert!(!is_command("running", "run"));
}

#[test]
fn an_empty_name_is_not_a_command() {
    assert!(!is_command("", ""));
    assert!(!is_command("run", ""));
}

#[test]
fn blank_text_is_not_a_command() {
    assert!(!is_command("   ", "run"));
}
