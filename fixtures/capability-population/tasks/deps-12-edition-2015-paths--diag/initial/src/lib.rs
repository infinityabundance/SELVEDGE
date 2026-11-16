//! A textkit label. This does not build under edition 2015.

use textkit::label;

pub fn label_of() -> &'static str {
    label()
}
