use trio_03_logic_quantifier::{repair_path, Premise, Quantifier, Repair};

const ALL: Quantifier = Quantifier::All;
const SOME: Quantifier = Quantifier::Some;

fn premise(subject: &'static str, quantifier: Quantifier, object: &'static str) -> Premise {
    Premise { subject, quantifier, object }
}

#[test]
fn a_direct_all_premise_needs_no_strengthening() {
    let premises = [premise("a", ALL, "b")];
    assert_eq!(repair_path(&premises, "a", "b"),
               Some(Repair { path: vec![0], strengthened: vec![] }));
}

#[test]
fn a_two_step_chain_is_found() {
    let premises = [premise("a", ALL, "b"), premise("b", ALL, "c")];
    assert_eq!(repair_path(&premises, "a", "c"),
               Some(Repair { path: vec![0, 1], strengthened: vec![] }));
}

#[test]
fn a_some_link_must_be_strengthened() {
    let premises = [premise("a", SOME, "b"), premise("b", ALL, "c")];
    assert_eq!(repair_path(&premises, "a", "c"),
               Some(Repair { path: vec![0, 1], strengthened: vec![0] }));
}

#[test]
fn the_shortest_chain_wins() {
    let premises = [
        premise("a", ALL, "x"),
        premise("x", ALL, "z"),
        premise("a", ALL, "z"),
    ];
    assert_eq!(repair_path(&premises, "a", "z"),
               Some(Repair { path: vec![2], strengthened: vec![] }));
}

#[test]
fn identical_nodes_need_no_premise() {
    let premises: [Premise; 0] = [];
    assert_eq!(repair_path(&premises, "a", "a"),
               Some(Repair { path: vec![], strengthened: vec![] }));
}

#[test]
fn an_unreachable_target_is_none() {
    let premises = [premise("a", ALL, "b")];
    assert_eq!(repair_path(&premises, "a", "zz"), None);
}
