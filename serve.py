#!/usr/bin/env python3
"""
Utility script for serving the release APK and pulling debug images from mobile.

Usage:
    python serve.py              # Serve the release APK on port 8080
    python serve.py pull         # Pull debug inputs from device via adb and render as PNGs
    python serve.py pull --clean # Pull debug inputs and clear them from the device
"""
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APK_DIR = PROJECT_ROOT / "mobile" / "sensor-app" / "android" / "app" / "build" / "outputs" / "apk" / "release"
DEBUG_OUT = PROJECT_ROOT / "mobile" / "sensor-app" / "model_training" / "debugging" / "debug_pngs"
PACKAGE = "com.anonymous.sensorapp"


def serve_apk():
    os.chdir(APK_DIR)
    SimpleHTTPRequestHandler.extensions_map['.apk'] = 'application/vnd.android.package-archive'
    print(f"Serving APK from {APK_DIR} on port 8080...")
    HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler).serve_forever()


def pull_debug(clean: bool = False):
    """Pull debug_inputs from device via adb, convert JSON tensors to PNGs."""
    # Check adb is available
    try:
        subprocess.run(["adb", "devices"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: adb not found in PATH")
        sys.exit(1)

    # List files on device
    result = subprocess.run(
        ["adb", "shell", f"run-as {PACKAGE} ls files/debug_inputs"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("No debug_inputs directory found on device (toggle Debug: ON in the app first).")
        return

    files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    if not files:
        print("debug_inputs directory is empty.")
        return

    print(f"Found {len(files)} debug files on device.")

    # Pull via tar
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "debug_inputs.tar")
        with open(tar_path, "wb") as f:
            subprocess.run(
                ["adb", "exec-out", f"run-as {PACKAGE} tar -C files -cf - debug_inputs"],
                stdout=f, check=True,
            )

        with tarfile.open(tar_path) as tar:
            tar.extractall(tmpdir)

        json_dir = os.path.join(tmpdir, "debug_inputs")
        if not os.path.isdir(json_dir):
            print("Error: tar extraction failed.")
            return

        json_files = sorted(Path(json_dir).glob("*.json"))
        print(f"Converting {len(json_files)} JSON tensors to PNGs...")

        DEBUG_OUT.mkdir(parents=True, exist_ok=True)

        for jf in json_files:
            with open(jf) as f:
                tensor = json.load(f)

            w = tensor["width"]
            h = tensor["height"]
            data = tensor["data"]

            # Write as raw PPM (no Pillow dependency needed)
            png_name = jf.stem + ".ppm"
            out_path = DEBUG_OUT / png_name
            with open(out_path, "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                for i in range(0, len(data), 3):
                    r = max(0, min(255, int(data[i])))
                    g = max(0, min(255, int(data[i + 1])))
                    b = max(0, min(255, int(data[i + 2])))
                    f.write(struct.pack("BBB", r, g, b))
            print(f"  -> {out_path}")

        # Try converting to PNG if Pillow is available
        try:
            from PIL import Image
            for ppm in DEBUG_OUT.glob("*.ppm"):
                png_path = ppm.with_suffix(".png")
                Image.open(ppm).save(png_path)
                ppm.unlink()
                print(f"  Converted to PNG: {png_path.name}")
        except ImportError:
            print("  (Pillow not installed — saved as PPM. Install Pillow for PNG: pip install Pillow)")

    print(f"\nDebug images saved to: {DEBUG_OUT}")

    if clean:
        subprocess.run(
            ["adb", "shell", f"run-as {PACKAGE} rm -rf files/debug_inputs"],
            check=True,
        )
        print("Cleared debug_inputs from device.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pull":
        clean = "--clean" in sys.argv
        pull_debug(clean=clean)
    else:
        serve_apk()
