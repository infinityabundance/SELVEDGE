use trio_04_invariant_ledger::Ledger;

#[test]
fn deposit_updates_both_components() {
    let mut ledger = Ledger::new();
    ledger.deposit(10);
    assert_eq!(ledger.total(), 10);
    assert_eq!(ledger.balance(), 10);
}

#[test]
fn withdraw_updates_both_components() {
    let mut ledger = Ledger::new();
    ledger.deposit(10);
    ledger.withdraw(4).unwrap();
    assert_eq!(ledger.total(), 6);
    assert_eq!(ledger.balance(), 6);
    assert_eq!(ledger.entries(), &[10, -4]);
}

#[test]
fn overdraw_is_rejected_and_leaves_state_unchanged() {
    let mut ledger = Ledger::new();
    ledger.deposit(5);
    assert_eq!(ledger.withdraw(6), Err("insufficient funds".to_string()));
    assert_eq!(ledger.total(), 5);
    assert_eq!(ledger.balance(), 5);
    assert_eq!(ledger.entries(), &[5]);
}

#[test]
fn invalid_amount_is_rejected_and_leaves_state_unchanged() {
    let mut ledger = Ledger::new();
    ledger.deposit(5);
    assert_eq!(ledger.withdraw(0), Err("invalid amount".to_string()));
    assert_eq!(ledger.withdraw(-1), Err("invalid amount".to_string()));
    assert_eq!(ledger.total(), 5);
    assert_eq!(ledger.entries(), &[5]);
}

#[test]
fn the_invariant_holds_after_a_sequence() {
    let mut ledger = Ledger::new();
    ledger.deposit(10);
    ledger.withdraw(3).unwrap();
    ledger.deposit(2);
    assert_eq!(ledger.total(), ledger.balance());
    assert_eq!(ledger.total(), 9);
}
