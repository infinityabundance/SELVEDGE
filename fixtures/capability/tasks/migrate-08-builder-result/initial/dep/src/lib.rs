//! Settings, version 2. Construction goes through a fallible builder.

#[derive(Debug)]
pub struct Config {
    pub level: u8,
}

#[derive(Debug)]
pub struct BuildError;

impl core::fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "level must be non-zero")
    }
}

pub struct Builder {
    level: u8,
}

impl Builder {
    pub fn level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    pub fn build(self) -> Result<Config, BuildError> {
        if self.level == 0 {
            Err(BuildError)
        } else {
            Ok(Config { level: self.level })
        }
    }
}

impl Config {
    pub fn builder() -> Builder {
        Builder { level: 1 }
    }
}
