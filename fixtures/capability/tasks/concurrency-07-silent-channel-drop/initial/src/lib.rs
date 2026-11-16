//! Send every value, currently discarding every error.

use std::sync::mpsc::SyncSender;

pub fn send_all(sender: &SyncSender<i32>, values: &[i32]) -> Result<(), String> {
    for value in values {
        let _ = sender.try_send(*value);
    }
    Ok(())
}
