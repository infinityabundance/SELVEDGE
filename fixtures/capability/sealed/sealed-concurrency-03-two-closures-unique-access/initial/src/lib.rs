//! Two scoped increments. The crate does not compile: both closures hold
//! the same mutable reference.

pub fn bump_both(counter: &mut u32) {
    std::thread::scope(|scope| {
        scope.spawn(|| {
            *counter += 1;
        });
        scope.spawn(|| {
            *counter += 1;
        });
    });
}
