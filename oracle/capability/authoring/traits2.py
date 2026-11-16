"""Trait and impl repairs, batch two (six tasks).

The failures are the real trait-system diagnostics: a missing item, an
unimplemented supertrait, a missing bound, a non-object-safe associated
constant, an overlapping `From` impl, and the orphan rule.
"""

from .common import code

TASKS = [
    {
        "id": "traits-04-missing-method",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Complete a trait impl that omits a required method.",
        "prompt_intro": (
            "This crate does not compile. `Square` implements `Shape` but omits the "
            "required `name` method. Implement it returning `\"square\"`. Keep the "
            "trait, the struct and `describe` unchanged."),
        "observed_failure": (
            "error[E0046]: not all trait items implemented, missing: `name`\n"
            " --> src/lib.rs:6:1\n"
            "  |\n"
            "6 | impl Shape for Square {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^ missing `name` in implementation"),
        "files": {
            "src/lib.rs": code("""
                //! Shapes. `Square` does not implement the full trait.

                pub trait Shape {
                    fn area(&self) -> f64;
                    fn name(&self) -> &'static str;
                }

                pub struct Square {
                    pub side: f64,
                }

                impl Shape for Square {
                    fn area(&self) -> f64 {
                        self.side * self.side
                    }
                }

                pub fn describe<T: Shape>(shape: &T) -> String {
                    format!("{}:{:.1}", shape.name(), shape.area())
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Shapes.

                pub trait Shape {
                    fn area(&self) -> f64;
                    fn name(&self) -> &'static str;
                }

                pub struct Square {
                    pub side: f64,
                }

                impl Shape for Square {
                    fn area(&self) -> f64 {
                        self.side * self.side
                    }

                    fn name(&self) -> &'static str {
                        "square"
                    }
                }

                pub fn describe<T: Shape>(shape: &T) -> String {
                    format!("{}:{:.1}", shape.name(), shape.area())
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "self.side * self.side",
                     "replace": "self.side + self.side"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0046"]},
    },
    {
        "id": "traits-05-supertrait-bound",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Implement a supertrait required by a new trait bound.",
        "prompt_intro": (
            "This crate does not compile. `Node` now requires `Debug`, but `Leaf` "
            "does not implement it. Derive the missing implementation; do not weaken "
            "the trait. Keep `summarise` and the signatures."),
        "observed_failure": (
            "error[E0277]: `Leaf` doesn't implement `Debug`\n"
            " --> src/lib.rs:4:18\n"
            "  |\n"
            "4 | pub trait Node: Debug {\n"
            "  |                  ^^^^^ `Leaf` cannot be formatted using `{:?}`\n"
            "  |\n"
            "  = help: the trait `Debug` is not implemented for `Leaf`"),
        "files": {
            "src/lib.rs": code("""
                //! Nodes that can be debug-formatted.

                use core::fmt::Debug;

                pub trait Node: Debug {
                    fn id(&self) -> u32;
                }

                pub struct Leaf {
                    pub id: u32,
                }

                impl Node for Leaf {
                    fn id(&self) -> u32 {
                        self.id
                    }
                }

                pub fn summarise(nodes: &[Box<dyn Node>]) -> String {
                    nodes
                        .iter()
                        .map(|node| format!("{:?}:{}", node, node.id()))
                        .collect::<Vec<_>>()
                        .join(",")
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_05_supertrait_bound::{summarise, Leaf, Node};

                #[test]
                fn reports_the_id() {
                    assert_eq!(Leaf { id: 2 }.id(), 2);
                }

                #[test]
                fn leaf_is_debug() {
                    assert!(format!("{:?}", Leaf { id: 7 }).contains("Leaf"));
                }

                #[test]
                fn summarises_a_boxed_node() {
                    let nodes: Vec<Box<dyn Node>> = vec![Box::new(Leaf { id: 4 })];
                    let text = summarise(&nodes);
                    assert!(text.contains("4"));
                    assert!(text.contains("Leaf"));
                }

                #[test]
                fn summarises_several_nodes() {
                    let nodes: Vec<Box<dyn Node>> = vec![
                        Box::new(Leaf { id: 1 }),
                        Box::new(Leaf { id: 2 }),
                    ];
                    assert_eq!(summarise(&nodes).matches("Leaf").count(), 2);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Nodes that can be debug-formatted.

                use core::fmt::Debug;

                pub trait Node: Debug {
                    fn id(&self) -> u32;
                }

                #[derive(Debug)]
                pub struct Leaf {
                    pub id: u32,
                }

                impl Node for Leaf {
                    fn id(&self) -> u32 {
                        self.id
                    }
                }

                pub fn summarise(nodes: &[Box<dyn Node>]) -> String {
                    nodes
                        .iter()
                        .map(|node| format!("{:?}:{}", node, node.id()))
                        .collect::<Vec<_>>()
                        .join(",")
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "#[derive(Debug)]",
                     "replace": "#[derive(Clone)]"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0277"]},
    },
    {
        "id": "traits-06-missing-generic-bound",
        "category": "trait-impl",
        "difficulty": "easy",
        "capability": "Add the bound that makes a generic helper call a trait method.",
        "prompt_intro": (
            "This crate does not compile. `total` calls `Measure::value` on a generic "
            "`T` without a bound. Add the bound; keep the signature's return type and "
            "the trait. `total` returns the sum of the values."),
        "observed_failure": (
            "error[E0599]: no method named `value` found for reference `&T` in the current scope\n"
            " --> src/lib.rs:11:33\n"
            "  |\n"
            "11 |     items.iter().map(|item| item.value()).sum()\n"
            "  |                                 ^^^^^ method not found in `&T`\n"
            "  |\n"
            "help: consider restricting type parameter `T`"),
        "files": {
            "src/lib.rs": code("""
                //! Summing measured values.

                pub trait Measure {
                    fn value(&self) -> u32;
                }

                pub struct Reading(pub u32);

                impl Measure for Reading {
                    fn value(&self) -> u32 {
                        self.0
                    }
                }

                pub fn total<T>(items: &[T]) -> u32 {
                    items.iter().map(|item| item.value()).sum()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_06_missing_generic_bound::{total, Measure, Reading};

                #[test]
                fn sums_measurements() {
                    assert_eq!(total(&[Reading(1), Reading(2), Reading(3)]), 6);
                }

                #[test]
                fn empty_is_zero() {
                    assert_eq!(total::<Reading>(&[]), 0);
                }

                #[test]
                fn works_with_another_measure() {
                    struct Weight(u32);
                    impl Measure for Weight {
                        fn value(&self) -> u32 { self.0 }
                    }
                    assert_eq!(total(&[Weight(5), Weight(7)]), 12);
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Summing measured values.

                pub trait Measure {
                    fn value(&self) -> u32;
                }

                pub struct Reading(pub u32);

                impl Measure for Reading {
                    fn value(&self) -> u32 {
                        self.0
                    }
                }

                pub fn total<T: Measure>(items: &[T]) -> u32 {
                    items.iter().map(|item| item.value()).sum()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "item.value()", "replace": "item.value() + 1"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0599"]},
    },
    {
        "id": "traits-07-associated-const-object-safety",
        "category": "trait-impl",
        "difficulty": "hard",
        "capability": "Make a trait object-safe by moving an associated constant to a method.",
        "prompt_intro": (
            "This crate does not compile. `Unit` has an associated constant, so it "
            "cannot be used as `Box<dyn Unit>`. Replace the constant with a `scale` "
            "method and give `convert` a default implementation `value * scale()`. "
            "Implement `scale` for both units: `Meters` is 1000 and `Centimeters` is "
            "10. Keep `convert_all`."),
        "observed_failure": (
            "error[E0038]: the trait `Unit` cannot be made into an object\n"
            " --> src/lib.rs:14:39\n"
            "  |\n"
            "14 | pub fn convert_all(units: &[Box<dyn Unit>], value: u32) -> Vec<u32> {\n"
            "  |                                       ^^^^ `Unit` cannot be made into an object\n"
            "  |\n"
            "  = note: the trait cannot be made into an object because it requires `Self: Sized`"),
        "files": {
            "src/lib.rs": code("""
                //! Units. The associated constant prevents dynamic dispatch.

                pub trait Unit {
                    const SCALE: u32;

                    fn convert(&self, value: u32) -> u32;
                }

                pub struct Meters;

                impl Unit for Meters {
                    const SCALE: u32 = 1000;

                    fn convert(&self, value: u32) -> u32 {
                        value * Self::SCALE
                    }
                }

                pub struct Centimeters;

                impl Unit for Centimeters {
                    const SCALE: u32 = 10;

                    fn convert(&self, value: u32) -> u32 {
                        value * Self::SCALE
                    }
                }

                pub fn convert_all(units: &[Box<dyn Unit>], value: u32) -> Vec<u32> {
                    units.iter().map(|unit| unit.convert(value)).collect()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
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
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Units.

                pub trait Unit {
                    fn scale(&self) -> u32;

                    fn convert(&self, value: u32) -> u32 {
                        value * self.scale()
                    }
                }

                pub struct Meters;

                impl Unit for Meters {
                    fn scale(&self) -> u32 {
                        1000
                    }
                }

                pub struct Centimeters;

                impl Unit for Centimeters {
                    fn scale(&self) -> u32 {
                        10
                    }
                }

                pub fn convert_all(units: &[Box<dyn Unit>], value: u32) -> Vec<u32> {
                    units.iter().map(|unit| unit.convert(value)).collect()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "1000", "replace": "999"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0038"]},
    },
    {
        "id": "traits-08-conflicting-from",
        "category": "trait-impl",
        "difficulty": "hard",
        "capability": "Resolve an overlapping blanket From impl by keeping concrete conversions.",
        "prompt_intro": (
            "This crate does not compile. A blanket `impl<T: Into<f64>> From<T> for "
            "Celsius` overlaps the concrete `f64` and `i32` conversions. Remove the "
            "blanket impl and keep both concrete conversions. Keep `Celsius` and the "
            "`From` conversions' behaviour."),
        "observed_failure": (
            "error[E0119]: conflicting implementations of trait `From<T>` for type `Celsius`\n"
            " --> src/lib.rs:9:1\n"
            "  |\n"
            "3 | impl From<f64> for Celsius {\n"
            "  | -------------------------- first implementation here\n"
            "...\n"
            "9 | impl<T: Into<f64>> From<T> for Celsius {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ conflicting implementation for `Celsius`"),
        "files": {
            "src/lib.rs": code("""
                //! Temperature conversion with an overlapping blanket impl.

                #[derive(Debug, PartialEq)]
                pub struct Celsius(pub f64);

                impl From<f64> for Celsius {
                    fn from(value: f64) -> Self {
                        Celsius(value)
                    }
                }

                impl From<i32> for Celsius {
                    fn from(value: i32) -> Self {
                        Celsius(value as f64)
                    }
                }

                impl<T: Into<f64>> From<T> for Celsius {
                    fn from(value: T) -> Self {
                        Celsius(value.into())
                    }
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_08_conflicting_from::Celsius;

                #[test]
                fn converts_from_f64() {
                    assert_eq!(Celsius::from(21.0), Celsius(21.0));
                }

                #[test]
                fn converts_from_i32() {
                    assert_eq!(Celsius::from(7), Celsius(7.0));
                }

                #[test]
                fn converts_negative_i32() {
                    assert_eq!(Celsius::from(-3), Celsius(-3.0));
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Temperature conversion.

                #[derive(Debug, PartialEq)]
                pub struct Celsius(pub f64);

                impl From<f64> for Celsius {
                    fn from(value: f64) -> Self {
                        Celsius(value)
                    }
                }

                impl From<i32> for Celsius {
                    fn from(value: i32) -> Self {
                        Celsius(value as f64)
                    }
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": "Celsius(value as f64)",
                     "replace": "Celsius(value as f64 + 1.0)"},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0119"]},
    },
    {
        "id": "traits-09-orphan-rule",
        "category": "trait-impl",
        "difficulty": "medium",
        "capability": "Respect the orphan rule with a local newtype wrapper.",
        "prompt_intro": (
            "This crate does not compile. It implements `Display` directly for "
            "`Vec<u8>`, which the orphan rule forbids. Introduce a local newtype "
            "`Bytes(pub Vec<u8>)`, implement `Display` for it (comma-separated "
            "decimal bytes), and keep `pub fn render(values: &[u8]) -> String`."),
        "observed_failure": (
            "error[E0117]: only traits defined in the current crate can be implemented for types defined outside of the crate\n"
            " --> src/lib.rs:1:1\n"
            "  |\n"
            "1 | impl core::fmt::Display for Vec<u8> {\n"
            "  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
            "  | `Vec<u8>` is not defined in the current crate"),
        "files": {
            "src/lib.rs": code("""
                //! Render bytes as decimal text. The Display impl on Vec<u8> violates
                //! the orphan rule.

                impl core::fmt::Display for Vec<u8> {
                    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        let text = self
                            .iter()
                            .map(|byte| byte.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        write!(formatter, "{text}")
                    }
                }

                pub fn render(values: &[u8]) -> String {
                    values.to_vec().to_string()
                }
            """),
        },
        "hidden": {
            "tests/check.rs": code("""
                use traits_09_orphan_rule::render;

                #[test]
                fn joins_decimal_bytes() {
                    assert_eq!(render(&[1, 2, 3]), "1,2,3");
                }

                #[test]
                fn empty_is_empty() {
                    assert_eq!(render(&[]), "");
                }

                #[test]
                fn single_byte_has_no_separator() {
                    assert_eq!(render(&[255]), "255");
                }

                #[test]
                fn zeros_are_rendered() {
                    assert_eq!(render(&[0, 0]), "0,0");
                }
            """),
        },
        "reference": {
            "src/lib.rs": code("""
                //! Render bytes as decimal text through a local newtype.

                pub struct Bytes(pub Vec<u8>);

                impl core::fmt::Display for Bytes {
                    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                        let text = self
                            .0
                            .iter()
                            .map(|byte| byte.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        write!(formatter, "{text}")
                    }
                }

                pub fn render(values: &[u8]) -> String {
                    Bytes(values.to_vec()).to_string()
                }
            """),
        },
        "mutation": {"file": "src/lib.rs", "find": '.join(",")', "replace": '.join("-")'},
        "initial_failure": {"kind": "compile_error", "must_contain": ["E0117"]},
    },
]
