use repo_01_lifetime_propagation::{cheapest_name, report, store, Item};

fn item(name: &str, price: u32) -> Item {
    Item { name: name.to_string(), price }
}

#[test]
fn names_the_cheapest() {
    let items = vec![item("b", 5), item("a", 2)];
    let fallback = item("z", 99);
    assert_eq!(cheapest_name(&items, &fallback), "a");
}

#[test]
fn uses_the_fallback_when_empty() {
    let items: Vec<Item> = vec![];
    let fallback = item("z", 99);
    assert_eq!(cheapest_name(&items, &fallback), "z");
}

#[test]
fn describes_the_cheapest() {
    let items = vec![item("b", 5)];
    let fallback = item("z", 99);
    assert_eq!(report::describe_cheapest(&items, &fallback), "b costs 5");
}

#[test]
fn store_finds_by_name() {
    let items = vec![item("b", 5)];
    assert_eq!(store::find(&items, "b").map(|item| item.price), Some(5));
}
