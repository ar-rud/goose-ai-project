# helper script to paint gray scale mask according to goose 

import os
import numpy as np
from PIL import Image

# 1. Load the pitch-black single-channel PNG mask


## TO CHANGE DIRS PATHES BEFORE USE
mask_path = "./predictions/2022-07-22_flight__0080_1658494248904312093_prediction.png"
mask = Image.open(mask_path)
mask_data = np.array(mask)

# 2. Define exact GOOSE dataset RGB Color Palette (12 classes)
palette = np.array([
    [127, 150, 143], # 0: vegetation (Muted gray-green)
    [255, 20, 126],  # 1: terrain (Hot pink)
    [220, 140, 126], # 2: sky (Salmon/peach)
    [158, 122, 186], # 3: construction (Muted purple)
    [0, 75, 73],     # 4: vehicle (Dark teal)
    [162, 0, 75],    # 5: road (Dark burgundy/magenta)
    [209, 216, 0],   # 6: object (Chartreuse/acid green)
    [255, 212, 234], # 7: void (Light pink)
    [182, 191, 84],  # 8: sign (Olive/khaki)
    [138, 168, 255], # 9: human (Periwinkle/light blue)
    [148, 162, 185], # 10: water (Gray-blue)
    [74, 62, 101],   # 11: animal (Dark navy/purple)
], dtype=np.uint8)

# 3. Map class indices (0-11) directly to their RGB colors
# np.clip ensures index errors don't occur if unexpected pixel values exist
colorized = palette[np.clip(mask_data, 0, 11)]

# 4. Save as a standard RGB PNG image
base_name = os.path.splitext(os.path.basename(mask_path))[0]
output_path = f"./{base_name}_mask_colored.png"

Image.fromarray(colorized).save(output_path)
print(f"Saved colored mask to {output_path}")