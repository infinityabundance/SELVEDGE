// A storage scope is not evidence of semantic exactness. Passing a scope where
// a witness is required is a type error, so a correct scoped read cannot be
// promoted into an exact neural-execution claim.
//@ expect: expected `NativeSetWitness`, found `ExecutionScope`
use selvedge_core::{ExactSelection, SelectionDecision, SelectionProvenance};
use selvedge_store::ExecutionScope;

fn promote(decision: SelectionDecision, scope: ExecutionScope) -> ExactSelection {
    ExactSelection::from_witness(decision, scope).unwrap()
}

fn main() {
    let provenance = SelectionProvenance::approximate("predictor", Vec::new()).unwrap();
    let decision = SelectionDecision::new(Vec::new(), provenance).unwrap();
    let _ = promote;
    let _ = decision;
}
