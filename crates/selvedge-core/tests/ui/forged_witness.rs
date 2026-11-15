// A witness cannot be forged from a struct literal: every field is private and
// the only constructor compares the executed set with the declared native set.
//@ expect: private field
use selvedge_core::NativeSetWitness;

fn main() {
    let _forged = NativeSetWitness {
        bundle_id: "victim-bundle".to_owned(),
        declared_digest: selvedge_core::Sha256Digest::of(b"native"),
        executed_digest: selvedge_core::Sha256Digest::of(b"substring"),
        shard_count: 1,
    };
}
