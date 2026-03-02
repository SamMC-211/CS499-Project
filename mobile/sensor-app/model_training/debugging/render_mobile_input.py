import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

inp_dir = Path("debug_inputs")  # folder where you copied JSON files
out_dir = Path("debug_pngs")
out_dir.mkdir(exist_ok=True)

for jf in inp_dir.glob("*.json"):
    obj = json.loads(jf.read_text()) # stores loaded json object
    w, h, c = obj["width"], obj["height"], obj["channels"] #extract metadata from json
    arr = np.array(obj["data"], dtype=np.float32).reshape(h, w, c) # convert raw flaot data into numpy array (Float32Array)

    # handle either [0,1] or [0,255] range (Normalization handling)
    # if arr.max() <= 1.0:
    #     arr = arr * 255.0
    # img = np.clip(arr, 0, 255).astype(np.uint8)

    p99 = np.percentile(arr, 99)  # robust against a few 255 dots
    if p99 <= 1.5:
        vis = arr * 255.0
    else:
        vis = arr

    img = np.clip(vis, 0, 255).astype(np.uint8)

    out_path = out_dir / (jf.stem + ".png") # create output filename
    plt.imsave(out_path, img) # save image
    print("saved", out_path)