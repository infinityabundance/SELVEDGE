//! Configure a value.

pub fn configured(level: u8) -> Result<u8, String> {
    settings::Config::builder()
        .level(level)
        .build()
        .map(|config| config.level)
        .map_err(|_| "invalid level".to_string())
}
