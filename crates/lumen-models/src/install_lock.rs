//! Cross-process install mutex (contract §6).
//!
//! All Lumen apps installing into the same shared models root must hold this
//! OS-level exclusive file lock before downloading; the OS releases it
//! automatically if the process exits or crashes. The lock file itself may be
//! left behind. An in-process atomic flag is *not* a substitute — it only
//! debounces buttons within one app.

use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

pub const SENSEVOICE_INSTALL_LOCK_NAME: &str = ".sensevoice-install.lock";

pub struct ModelInstallLock {
    file: File,
}

impl ModelInstallLock {
    /// Try to acquire the exclusive install lock under `models_root`.
    ///
    /// Returns `Ok(None)` when another process holds the lock; callers should
    /// wait (allowing user cancellation) instead of starting a second
    /// download.
    pub fn try_acquire(models_root: &Path) -> io::Result<Option<Self>> {
        std::fs::create_dir_all(models_root)?;
        let file = match OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(models_root.join(SENSEVOICE_INSTALL_LOCK_NAME))
        {
            Ok(file) => file,
            Err(error) if is_lock_contended(&error) => return Ok(None),
            Err(error) => return Err(error),
        };
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(Self { file })),
            Err(error) if is_lock_contended(&error) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

fn is_lock_contended(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock
        || (cfg!(windows) && matches!(error.raw_os_error(), Some(32 | 33)))
}

impl Drop for ModelInstallLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::ModelInstallLock;
    use crate::test_support::temp_dir;

    #[test]
    fn only_one_installer_can_lock_a_shared_models_root() {
        let root = temp_dir("install-lock");

        let first = ModelInstallLock::try_acquire(&root).unwrap().unwrap();
        assert!(ModelInstallLock::try_acquire(&root).unwrap().is_none());
        drop(first);
        assert!(ModelInstallLock::try_acquire(&root).unwrap().is_some());

        let _ = std::fs::remove_dir_all(root);
    }
}
