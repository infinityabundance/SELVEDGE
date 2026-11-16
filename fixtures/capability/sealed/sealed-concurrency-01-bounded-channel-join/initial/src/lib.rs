//! A bounded channel drained too late. The crate builds, but the
//! producer is joined before the receiver drains.

use std::sync::mpsc::sync_channel;

pub fn collect_after_join(count: usize) -> Vec<usize> {
    let (sender, receiver) = sync_channel(1);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for value in 0..count {
                sender.send(value).unwrap();
            }
        });
    });
    receiver.iter().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn visible_three_messages() {
        assert_eq!(super::collect_after_join(3), vec![0, 1, 2]);
    }
}
