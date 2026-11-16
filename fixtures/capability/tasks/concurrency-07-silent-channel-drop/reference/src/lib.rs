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
