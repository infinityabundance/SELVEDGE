//! The application error.

use crate::parse::ParseError;

#[derive(Debug, PartialEq)]
pub enum AppError {
    Missing,
    Parse(usize),
    Io,
}

impl core::fmt::Display for AppError {
fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AppError::Missing => write!(formatter, "missing"),
            AppError::Parse(field) => write!(formatter, "malformed field {field}"),
            AppError::Io => write!(formatter, "io error"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<ParseError> for AppError {
    fn from(error: ParseError) -> Self {
        AppError::Parse(error.field)
    }
}

impl From<std::io::Error> for AppError {
    fn from(_error: std::io::Error) -> Self {
        AppError::Io
    }
}

#[cfg(test)]
mod visible {
    use crate::{error::AppError, service::import};

    #[test]
    fn imports_rows() {
        let path = std::env::temp_dir().join("repo-02-rows.txt");
        let lines = vec!["a,1".to_string(), "b,2".to_string()];
        assert_eq!(import(path.to_str().unwrap(), &lines), Ok(2));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a,1\nb,2\n");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reports_the_parse_field() {
        let path = std::env::temp_dir().join("repo-02-unused.txt");
        let lines = vec!["a,1".to_string(), "b,x".to_string()];
        assert_eq!(import(path.to_str().unwrap(), &lines), Err(AppError::Parse(1)));
    }

    #[test]
    fn reports_io_errors() {
        let lines = vec!["a,1".to_string()];
        assert!(matches!(import("/nonexistent-dir/rows.txt", &lines), Err(AppError::Io)));
    }
}
