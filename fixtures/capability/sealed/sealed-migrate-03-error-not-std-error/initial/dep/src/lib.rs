//! Parse, version 2. The error type no longer implements
//! `std::error::Error`; it renders itself only through `Display`.

pub struct Error {
    pub reason: &'static str,
}

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}", self.reason)
    }
}

pub fn parse(text: &str) -> Result<i64, Error> {
    text.trim()
        .parse()
        .map_err(|_| Error { reason: "not a number" })
}
