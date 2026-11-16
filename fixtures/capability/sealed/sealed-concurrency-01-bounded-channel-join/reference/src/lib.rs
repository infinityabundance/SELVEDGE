//! A bounded channel drained while the producer runs.

use std::sync::mpsc::sync_channel;

pub fn collect_after_join(count: usize) -> Vec<usize> {
    let (sender, receiver) = sync_channel(1);
    let mut out = Vec::new();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for value in 0..count {
                sender.send(value).unwrap();
            }
        });
        out.extend(receiver.iter());
    });
    out
}
