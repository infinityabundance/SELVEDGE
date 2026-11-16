use traits_02_coherence::{summary_of_text, Report, Summary};

#[test]
fn report_summary() {
    assert_eq!(Report("q3".to_string()).summary(), "report: q3");
}

#[test]
fn text_summary() {
    assert_eq!(summary_of_text("hello"), "text: hello");
}

#[test]
fn empty_text_summary() {
    assert_eq!(summary_of_text(""), "text: ");
}
