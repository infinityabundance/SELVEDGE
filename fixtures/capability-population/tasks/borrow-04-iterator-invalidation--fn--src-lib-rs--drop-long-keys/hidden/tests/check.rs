use borrow_04_iterator_invalidation::drop_long_keys;
use std::collections::BTreeMap;

fn map() -> BTreeMap<String, i32> {
    let mut map = BTreeMap::new();
    map.insert("a".to_string(), 1);
    map.insert("bb".to_string(), 2);
    map.insert("ccc".to_string(), 3);
    map
}

#[test]
fn removes_only_the_long_keys() {
    let mut map = map();
    assert_eq!(drop_long_keys(&mut map, 1), 2);
    assert_eq!(map.keys().cloned().collect::<Vec<_>>(), vec!["a".to_string()]);
}

#[test]
fn keeps_everything_when_the_bound_is_large() {
    let mut map = map();
    assert_eq!(drop_long_keys(&mut map, 10), 0);
    assert_eq!(map.len(), 3);
}

#[test]
fn empty_map_is_unchanged() {
    let mut map = BTreeMap::new();
    assert_eq!(drop_long_keys(&mut map, 0), 0);
}
