//! A codec label. This does not build: the import name is not the
//! library target name.

pub fn codec_label() -> &'static str {
    codec::label()
}
