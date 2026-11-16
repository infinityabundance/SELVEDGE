//! A codec checksum. This does not build: the import name is not the
//! library target name.

pub fn codec_total(text: &str) -> u32 {
    codec::total(text)
}
