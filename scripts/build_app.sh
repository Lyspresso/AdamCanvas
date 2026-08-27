#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_DIR=${SCRIPT_DIR:h}
APP_DIR="$PROJECT_DIR/build/Adam.app"
CONTENTS_DIR="$APP_DIR/Contents"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
ICON_SOURCE="$PROJECT_DIR/Resources/Adam.icon"
ICON_INFO="$PROJECT_DIR/build/Adam.icon-info.plist"
CEF_FRAMEWORK_NAME="Chromium Embedded Framework.framework"
HELPERS=(
    "Adam Helper"
    "Adam Helper (GPU)"
    "Adam Helper (Renderer)"
    "Adam Helper (Plugin)"
    "Adam Helper (Alerts)"
)

cd "$PROJECT_DIR"
cargo build --locked --release --bin Adam --bin adam-cef-helper

CEF_PACKAGE_VERSION=$(cargo metadata --locked --format-version 1 \
    | /usr/bin/jq -r '.packages[] | select(.name == "cef") | .version' \
    | /usr/bin/head -n 1)
CEF_RUNTIME_VERSION=${CEF_PACKAGE_VERSION#*+}
case "$(uname -m)" in
    arm64) CEF_ARCH=aarch64 ;;
    x86_64) CEF_ARCH=x86_64 ;;
    *) echo "Unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
esac
CEF_PLATFORM_DIR="cef_macos_$CEF_ARCH"
CEF_DIR=""

if [[ -n "${CEF_PATH:-}" ]]; then
    for candidate in \
        "$CEF_PATH/$CEF_RUNTIME_VERSION/$CEF_PLATFORM_DIR" \
        "$CEF_PATH"; do
        if [[ -d "$candidate/$CEF_FRAMEWORK_NAME" ]]; then
            CEF_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$CEF_DIR" ]]; then
    for candidate in "$PROJECT_DIR"/target/release/build/cef-dll-sys-*/out/"$CEF_PLATFORM_DIR"(N); do
        if [[ -d "$candidate/$CEF_FRAMEWORK_NAME" ]]; then
            CEF_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$CEF_DIR" ]]; then
    echo "Could not locate the CEF runtime produced by cef-dll-sys." >&2
    echo "Set CEF_PATH to the pinned CEF directory and rebuild." >&2
    exit 1
fi

rm -rf "$APP_DIR"
mkdir -p "$CONTENTS_DIR/MacOS" "$CONTENTS_DIR/Resources" "$FRAMEWORKS_DIR"
cp "$PROJECT_DIR/target/release/Adam" "$CONTENTS_DIR/MacOS/Adam"
cp "$PROJECT_DIR/Resources/Info.plist" "$CONTENTS_DIR/Info.plist"
cp "$PROJECT_DIR/Resources/Fonts/SourceSans3-LICENSE.md" "$CONTENTS_DIR/Resources/"
/usr/bin/ditto "$CEF_DIR/$CEF_FRAMEWORK_NAME" "$FRAMEWORKS_DIR/$CEF_FRAMEWORK_NAME"

for helper in "${HELPERS[@]}"; do
    HELPER_APP="$FRAMEWORKS_DIR/$helper.app"
    mkdir -p "$HELPER_APP/Contents/MacOS"
    cp "$PROJECT_DIR/target/release/adam-cef-helper" "$HELPER_APP/Contents/MacOS/$helper"
    cp "$PROJECT_DIR/Resources/Info.plist" "$HELPER_APP/Contents/Info.plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable $helper" "$HELPER_APP/Contents/Info.plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleName $helper" "$HELPER_APP/Contents/Info.plist"
    /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $helper" "$HELPER_APP/Contents/Info.plist"
    HELPER_ID=${helper:l}
    HELPER_ID=${HELPER_ID// /-}
    HELPER_ID=${HELPER_ID//\(/}
    HELPER_ID=${HELPER_ID//\)/}
    /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.lyspressopro.adam.$HELPER_ID" "$HELPER_APP/Contents/Info.plist"
    /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$HELPER_APP/Contents/Info.plist"
done

ICON_COMPILED=false
if [[ -d "$ICON_SOURCE" ]]; then
    if xcrun actool \
        --compile "$CONTENTS_DIR/Resources" \
        --platform macosx \
        --minimum-deployment-target 13.0 \
        --app-icon Adam \
        --output-partial-info-plist "$ICON_INFO" \
        "$ICON_SOURCE"; then
        /usr/libexec/PlistBuddy -c "Merge '$ICON_INFO'" "$CONTENTS_DIR/Info.plist"
        ICON_COMPILED=true
    else
        echo "Adam.icon compilation failed; using the checked-in Adam.icns fallback." >&2
    fi
fi
if [[ "$ICON_COMPILED" == false && -f "$PROJECT_DIR/Resources/Adam.icns" ]]; then
    cp "$PROJECT_DIR/Resources/Adam.icns" "$CONTENTS_DIR/Resources/Adam.icns"
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string Adam" "$CONTENTS_DIR/Info.plist" 2>/dev/null \
        || /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile Adam" "$CONTENTS_DIR/Info.plist"
fi

codesign --force --deep --timestamp=none --sign - "$FRAMEWORKS_DIR/$CEF_FRAMEWORK_NAME"
for helper in "${HELPERS[@]}"; do
    codesign --force --timestamp=none --sign - "$FRAMEWORKS_DIR/$helper.app"
done
codesign --force --timestamp=none --sign - "$APP_DIR"
codesign --verify --deep --strict "$APP_DIR"
echo "$APP_DIR"
