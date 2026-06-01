#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$PROJECT_DIR/weights"

FILE_ID="1icLjmSpTdEAA9kEW3BWHnAYv-hsXMYxS"
OUTPUT="$WEIGHTS_DIR/triad_swinb_simmim.pth"

mkdir -p "$WEIGHTS_DIR"

echo "[download_triad] Target: $OUTPUT"
echo "[download_triad] File ID: $FILE_ID"

if command -v gdown &>/dev/null; then
    echo "[download_triad] Using gdown..."
    gdown --id "$FILE_ID" -O "$OUTPUT"
else
    echo "[download_triad] gdown not found, using Python stdlib..."

    python3 - "$FILE_ID" "$OUTPUT" << 'PYEOF'
import sys
import urllib.request
import urllib.error

FILE_ID = sys.argv[1]
OUTPUT = sys.argv[2]

def _get_confirm_token(response):
    for key, value in response.info().items():
        if key.lower().startswith("download_warning"):
            return value
    return None

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = urllib.request.build_opener()
    session.addheaders = [("User-Agent", "Mozilla/5.0")]

    try:
        response = session.open(f"{URL}&id={file_id}")
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)

    token = _get_confirm_token(response)
    confirm_url = f"{URL}&confirm={token}&id={file_id}" if token else None

    if confirm_url:
        response = session.open(confirm_url)

    total = int(response.info().get("Content-Length", 0))
    downloaded = 0
    chunk_size = 32768
    with open(destination, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                print(f"\r  Progress: {downloaded}/{total} bytes ({pct:.1f}%)", end="")
    if total:
        print()

download_file_from_google_drive(FILE_ID, OUTPUT)
print("  Done.")
PYEOF
fi

if [ -f "$OUTPUT" ]; then
    SIZE=$(wc -c < "$OUTPUT" | tr -d ' ')
    echo "[download_triad] Success: $OUTPUT ($SIZE bytes)"
else
    echo "[download_triad] ERROR: file was not created." >&2
    exit 1
fi
