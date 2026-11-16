//! Summaries for a report and for text.

use core::fmt::Display;

pub trait Summary {
    fn summary(&self) -> String;
}

#[derive(Debug)]
pub struct Report(pub String);

impl Display for Report {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.0)
    }
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
    todo!()
}

#[cfg(test)]
mod visible {
    use super::*;

    #[test]
    fn report_summary() {
        assert_eq!(Report("q3".to_string()).summary(), "report: q3");
    }
}
