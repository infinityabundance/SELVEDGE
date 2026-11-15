// A storage scope is not evidence of semantic exactness. Passing a scope where
// a witness is required is a type error, so a correct scoped read cannot be
// promoted into an exact neural-execution claim.
//@ expect: expected `NativeSetWitness`, found `ExecutionScope`
use selvedge_core::{ExactSelection, NativeSetWitness, SelectionProvenance, SelectionDecision};

struct ExecutionScope {
    ids: Vec<u64>,
}

fn promote(decision: SelectionDecision, scope: ExecutionScope) -> ExactSelection {
    let _ = NativeSetWitness::compare;
    ExactSelection::from_witness(decision, scope).unwrap()
}

fn main() {
    let provenance = SelectionProvenance::approximate("predictor", Vec::new()).unwrap();
    let decision = SelectionDecision::new(Vec::new(), provenance).unwrap();
    let scope = ExecutionScope { ids: vec![1] };
    let _ = promote(decision, scope);
}
