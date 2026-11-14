#!/bin/sh
# Wrap the PyInstaller binary in a macOS .app bundle.
#
#   packaging/make_app.sh dist/ciu dist/CIU.app
#
# Why a wrapper rather than PyInstaller's own BUNDLE(): a windowed .app has no
# console. CIU's console is not decoration. It is the log the first-run
# instructions tell people to read when something fails, and closing it is how
# you stop the server. A windowed build would also be hard to quit, because
# uvicorn blocks the main thread and never reaches an event loop that could
# answer Cmd-Q.
#
# So the bundle's executable is a two-line script that asks Terminal to run
# the real binary. The user gets something that lives in Applications, appears
# in Spotlight and has a Dock icon, and still gets a window with the log in it
# that closes to stop.

set -eu

BIN="${1:-dist/ciu}"
APP="${2:-dist/CIU.app}"
VERSION="${3:-0.1.0}"

[ -f "$BIN" ] || { echo "no binary at $BIN. Build it first." >&2; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BIN" "$APP/Contents/Resources/ciu"
chmod +x "$APP/Contents/Resources/ciu"

cat > "$APP/Contents/MacOS/CIU" <<'LAUNCHER'
#!/bin/sh
# Hand the real binary to Terminal, so the user gets a window with the log.
DIR=$(cd "$(dirname "$0")/../Resources" && pwd)
exec open -a Terminal "$DIR/ciu"
LAUNCHER
chmod +x "$APP/Contents/MacOS/CIU"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>CIU</string>
  <key>CFBundleDisplayName</key>       <string>CIU</string>
  <key>CFBundleExecutable</key>        <string>CIU</string>
  <key>CFBundleIdentifier</key>        <string>com.runciu.ciu</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key>           <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>    <string>11.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <!-- Nothing here draws a window of its own, so it should not take focus
       from whatever the user was doing. Terminal takes focus instead. -->
  <key>LSBackgroundOnly</key>          <false/>
</dict>
</plist>
PLIST

echo "built $APP"
