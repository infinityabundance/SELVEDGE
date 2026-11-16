use deps_06_edition_unsafe_block::read_byte;

#[test]
fn reads_a_value() {
    let value = 42u8;
    assert_eq!(unsafe { read_byte(&value as *const u8) }, 42);
}

#[test]
fn reads_zero() {
    let value = 0u8;
    assert_eq!(unsafe { read_byte(&value as *const u8) }, 0);
}

#[test]
fn reads_the_maximum_byte() {
    let value = 255u8;
    assert_eq!(unsafe { read_byte(&value as *const u8) }, 255);
}
