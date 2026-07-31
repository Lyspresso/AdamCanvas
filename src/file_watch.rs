//! Notices when a tile's file changes on disk, so the canvas stays live.
//!
//! Adam's preview caches are already modification-time-aware on disk, but the
//! in-memory texture layer never revalidates: once a tile's preview is
//! resident it is returned forever. Saving a spreadsheet in Excel therefore
//! changed the file and Adam kept showing the old picture until restart.
//!
//! This watcher is a poll, not an FSEvents subscription, on purpose: it needs
//! no extra dependency or thread, one `stat` per file tile per tick is
//! microseconds, and a page holds tens of tiles, not thousands. The decision
//! logic is separated from the filesystem so it can be tested without one.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// How often the active page's file tiles are polled.
pub const POLL_INTERVAL: Duration = Duration::from_millis(900);

/// What one observation means for the tile that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// First sighting, or nothing new — leave the caches alone.
    Unchanged,
    /// The file's modification time moved — previews must be rebuilt.
    Changed,
}

#[derive(Default)]
pub struct FileWatch {
    seen: HashMap<Uuid, SystemTime>,
}

impl FileWatch {
    /// Feeds one `(tile, modification time)` observation.
    ///
    /// The first sighting of a tile is a baseline, never a change — otherwise
    /// every tile would rebuild its previews once at startup for nothing.
    ///
    /// A missing modification time (file absent or unreadable) keeps the old
    /// baseline rather than clearing it: applications that save atomically
    /// can briefly have no file at the path, and treating that window as a
    /// change would flash placeholders during every save. The change is
    /// reported when the file reappears with a new time.
    pub fn observe(&mut self, id: Uuid, mtime: Option<SystemTime>) -> Observation {
        let Some(mtime) = mtime else {
            return Observation::Unchanged;
        };
        match self.seen.insert(id, mtime) {
            Some(previous) if previous != mtime => Observation::Changed,
            Some(_) => Observation::Unchanged,
            None => Observation::Unchanged,
        }
    }

    /// Drops the baseline for a tile, so its next sighting starts fresh.
    pub fn forget(&mut self, id: Uuid) {
        self.seen.remove(&id);
    }
}

/// Stat helper the app glue uses; separated so tests never need it.
pub fn modification_time(path: &std::path::Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(seconds: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)
    }

    #[test]
    fn the_first_sighting_is_a_baseline_not_a_change() {
        let mut watch = FileWatch::default();
        let id = Uuid::new_v4();
        assert_eq!(watch.observe(id, Some(t(100))), Observation::Unchanged);
        // And staying put stays quiet.
        assert_eq!(watch.observe(id, Some(t(100))), Observation::Unchanged);
    }

    #[test]
    fn a_moved_modification_time_is_a_change_each_time_it_moves() {
        let mut watch = FileWatch::default();
        let id = Uuid::new_v4();
        watch.observe(id, Some(t(100)));
        assert_eq!(watch.observe(id, Some(t(101))), Observation::Changed);
        assert_eq!(watch.observe(id, Some(t(101))), Observation::Unchanged);
        assert_eq!(watch.observe(id, Some(t(102))), Observation::Changed);
        // Backwards also counts: a restored backup is still new content.
        assert_eq!(watch.observe(id, Some(t(50))), Observation::Changed);
    }

    #[test]
    fn a_vanished_file_keeps_its_baseline_through_an_atomic_save() {
        let mut watch = FileWatch::default();
        let id = Uuid::new_v4();
        watch.observe(id, Some(t(100)));
        // The atomic-save window: path briefly absent.
        assert_eq!(watch.observe(id, None), Observation::Unchanged);
        // Reappearing unchanged (save aborted) stays quiet...
        assert_eq!(watch.observe(id, Some(t(100))), Observation::Unchanged);
        watch.observe(id, None);
        // ...reappearing newer reports exactly one change.
        assert_eq!(watch.observe(id, Some(t(200))), Observation::Changed);
        assert_eq!(watch.observe(id, Some(t(200))), Observation::Unchanged);
    }

    #[test]
    fn tiles_do_not_interfere_and_forget_resets_the_baseline() {
        let mut watch = FileWatch::default();
        let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
        watch.observe(a, Some(t(100)));
        watch.observe(b, Some(t(500)));
        assert_eq!(watch.observe(a, Some(t(101))), Observation::Changed);
        assert_eq!(watch.observe(b, Some(t(500))), Observation::Unchanged);

        watch.forget(a);
        assert_eq!(
            watch.observe(a, Some(t(999))),
            Observation::Unchanged,
            "after forget, the next sighting is a baseline again"
        );
    }
}
