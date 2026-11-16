//! Summaries for a report and for text.

use core::fmt::Display;

pub trait Summary {
    fn summary(&self) -> String;
}

#[derive(Debug)]
pub struct Report(pub String);

impl Display for Report {
// TODO(FUNCTION)
}

impl Summary for Report {
    fn summary(&self) -> String {
        format!("report: {}", self.0)
    }
}

impl Summary for &str {
    fn summary(&self) -> String {
        format!("text: {self}")
    }
}

pub fn summary_of_text(text: &str) -> String {
    text.summary()
}

#[cfg(test)]
mod visible {
    use crate::{summary_of_text, Report, Summary};

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
}
