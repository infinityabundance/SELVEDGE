//! Mode selection, version 2. `fast` and `safe` are exclusive.

#[cfg(all(feature = "fast", feature = "safe"))]
compile_error!("features `fast` and `safe` are mutually exclusive");

pub fn mode() -> &'static str {
    #[cfg(feature = "fast")]
    {
        return "fast";
    }
    #[cfg(feature = "safe")]
    {
        return "safe";
    }
    "default"
}
