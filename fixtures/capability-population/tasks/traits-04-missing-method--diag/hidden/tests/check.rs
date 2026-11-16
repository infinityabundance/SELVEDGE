use traits_04_missing_method::{describe, Shape, Square};

#[test]
fn reports_the_area() {
    assert_eq!(Square { side: 3.0 }.area(), 9.0);
}

#[test]
fn reports_the_name() {
    assert_eq!(Square { side: 3.0 }.name(), "square");
}

#[test]
fn describes_a_shape() {
    assert_eq!(describe(&Square { side: 2.0 }), "square:4.0");
}

#[test]
fn works_with_another_implementor() {
    struct Triangle;
    impl Shape for Triangle {
        fn area(&self) -> f64 { 6.0 }
        fn name(&self) -> &'static str { "triangle" }
    }
    assert_eq!(describe(&Triangle), "triangle:6.0");
}
