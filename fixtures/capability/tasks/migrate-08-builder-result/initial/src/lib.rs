//! Configure a value. This does not build: version 2 removed the
//! direct constructor in favour of a fallible builder.

pub fn configured(level: u8) -> Result<u8, String> {
    let config = settings::Config::new(level);
    Ok(config.level)
}
