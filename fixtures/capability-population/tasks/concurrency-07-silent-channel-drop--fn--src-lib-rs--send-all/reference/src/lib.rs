//! Send every value and report saturation.

use std::sync::mpsc::{SyncSender, TrySendError};

pub fn send_all(sender: &SyncSender<i32>, values: &[i32]) -> Result<(), String> {
    for value in values {
        match sender.try_send(*value) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err("channel full".to_string())
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err("disconnected".to_string())
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod visible {
    use crate::send_all;
    use std::sync::mpsc::sync_channel;

    #[test]
    fn sends_when_there_is_room() {
        let (sender, receiver) = sync_channel(2);
        assert_eq!(send_all(&sender, &[1, 2]), Ok(()));
        assert_eq!(receiver.recv(), Ok(1));
        assert_eq!(receiver.recv(), Ok(2));
    }

    #[test]
    fn a_full_channel_is_an_error() {
        let (sender, _receiver) = sync_channel(1);
        sender.try_send(9).unwrap();
        assert_eq!(send_all(&sender, &[1]), Err("channel full".to_string()));
    }

    #[test]
    fn a_disconnected_channel_is_an_error() {
        let (sender, receiver) = sync_channel(1);
        drop(receiver);
        assert_eq!(send_all(&sender, &[1]), Err("disconnected".to_string()));
    }
}
