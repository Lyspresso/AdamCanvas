#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
APP_DIR="$PROJECT_DIR/build/Adam.app"
CONTENTS_DIR="$APP_DIR/Contents"
ICON_SOURCE="$PROJECT_DIR/Resources/Adam.icon"
ICON_INFO="$PROJECT_DIR/build/Adam.icon-info.plist"

cd "$PROJECT_DIR"
cargo build --release --bin Adam

rm -rf "$CONTENTS_DIR/Resources"
mkdir -p "$CONTENTS_DIR/MacOS" "$CONTENTS_DIR/Resources"
cp "$PROJECT_DIR/target/release/Adam" "$CONTENTS_DIR/MacOS/Adam"
cp "$PROJECT_DIR/Resources/Info.plist" "$CONTENTS_DIR/Info.plist"
cp "$PROJECT_DIR/Resources/Fonts/SourceSans3-LICENSE.md" "$CONTENTS_DIR/Resources/"

if [[ -d "$ICON_SOURCE" ]]; then
    xcrun actool \
        --compile "$CONTENTS_DIR/Resources" \
        --platform macosx \
        --minimum-deployment-target 13.0 \
        --app-icon Adam \
        --output-partial-info-plist "$ICON_INFO" \
        "$ICON_SOURCE"
    /usr/libexec/PlistBuddy -c "Merge '$ICON_INFO'" "$CONTENTS_DIR/Info.plist"
elif [[ -f "$PROJECT_DIR/Resources/Adam.icns" ]]; then
    cp "$PROJECT_DIR/Resources/Adam.icns" "$CONTENTS_DIR/Resources/Adam.icns"
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string Adam" "$CONTENTS_DIR/Info.plist" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile Adam" "$CONTENTS_DIR/Info.plist"
fi

codesign --force --timestamp=none --sign - "$APP_DIR"
echo "$APP_DIR"
