use std::path::Path;

#[cfg(target_os = "macos")]
mod imp {
    use objc2_app_kit::NSWorkspace;
    use objc2_foundation::{NSArray, NSString, NSURL};
    use std::{
        path::Path,
        process::{Command, Stdio},
        thread,
    };

    pub fn open_path(path: &Path) -> bool {
        let path = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&path);
        NSWorkspace::sharedWorkspace().openURL(&url)
    }

    pub fn open_url(value: &str) -> bool {
        let value = NSString::from_str(value);
        NSURL::URLWithString(&value).is_some_and(|url| NSWorkspace::sharedWorkspace().openURL(&url))
    }

    pub fn reveal(path: &Path) {
        let path = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&path);
        let urls = NSArray::from_retained_slice(&[url]);
        NSWorkspace::sharedWorkspace().activateFileViewerSelectingURLs(&urls);
    }

    pub fn quick_look(path: &Path) {
        let path = path.to_owned();
        let _ = thread::Builder::new()
            .name("adam-quick-look".into())
            .spawn(move || {
                let _ = Command::new("/usr/bin/qlmanage")
                    .arg("-p")
                    .arg(path)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            });
    }

    pub fn reduce_motion_enabled() -> bool {
        NSWorkspace::sharedWorkspace().accessibilityDisplayShouldReduceMotion()
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::path::Path;

    pub fn open_path(path: &Path) -> bool {
        open::that(path).is_ok()
    }

    pub fn open_url(value: &str) -> bool {
        open::that(value).is_ok()
    }

    pub fn reveal(path: &Path) {
        let _ = open::that(path.parent().unwrap_or(path));
    }

    pub fn quick_look(path: &Path) {
        let _ = open::that(path);
    }

    pub fn reduce_motion_enabled() -> bool {
        false
    }
}

pub fn open_path(path: &Path) -> bool {
    imp::open_path(path)
}

pub fn open_url(value: &str) -> bool {
    imp::open_url(value)
}

pub fn reveal(path: &Path) {
    imp::reveal(path);
}

pub fn quick_look(path: &Path) {
    imp::quick_look(path);
}

pub fn reduce_motion_enabled() -> bool {
    imp::reduce_motion_enabled()
}
