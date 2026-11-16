use traits_07_associated_const_object_safety::{convert_all, Centimeters, Meters, Unit};

#[test]
fn converts_meters() {
    assert_eq!(Meters.convert(2), 2000);
}

#[test]
fn converts_centimeters() {
    assert_eq!(Centimeters.convert(3), 30);
}

#[test]
fn converts_through_trait_objects() {
    let units: Vec<Box<dyn Unit>> = vec![Box::new(Meters), Box::new(Centimeters)];
    assert_eq!(convert_all(&units, 5), vec![5000, 50]);
}

#[test]
fn zero_converts_to_zero() {
    let units: Vec<Box<dyn Unit>> = vec![Box::new(Meters)];
    assert_eq!(convert_all(&units, 0), vec![0]);
}
