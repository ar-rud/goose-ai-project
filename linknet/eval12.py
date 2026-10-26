import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# Settings 
TEST_DIR = "small-goose/images/test"
WEIGHTS = "linknet_best_model.pth"
SIZE = (512, 256)          
NUM_CLASSES = 12
BATCH_SIZE = 4
OUT_DIR = "test_predictions"
SAVE_EVERY = 1            
CLASS_NAMES = [
    'vegetation', 'terrain', 'sky', 'construction',
    'vehicle', 'road', 'object', 'void', 'sign',
    'human', 'water', 'animal'
]

PALETTE = np.array([
    [34, 139, 34], [160, 120, 70], [135, 206, 235], [128, 128, 128],
    [220, 20, 60], [90, 90, 90], [255, 165, 0], [0, 0, 0],
    [255, 255, 0], [255, 0, 255], [0, 100, 255], [139, 69, 19],
], dtype=np.uint8)


class TestDataset(Dataset):
    def __init__(self, root, size):
        self.size = size
        self.root = root
        self.paths = []
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.paths.append(os.path.join(dirpath, f))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            img = np.asarray(im.convert("RGB").resize(self.size, Image.BILINEAR),
                             dtype=np.uint8).copy()
        return torch.from_numpy(img).permute(2, 0, 1), idx


def colorize(mask):
    return PALETTE[np.clip(mask, 0, NUM_CLASSES - 1)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "vis"), exist_ok=True)

    dataset = TestDataset(TEST_DIR, SIZE)
    print(f"Test images found: {len(dataset)}")
    if len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = smp.Linknet(encoder_name="mobilenet_v2", encoder_weights=None,
                        in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

    pixel_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

    with torch.inference_mode():
        for images, idxs in loader:
            x = (images.to(device).float() - mean) / std
            preds = model(x).argmax(dim=1).cpu()
            pixel_counts += torch.bincount(preds.flatten(), minlength=NUM_CLASSES)

            for i in range(len(idxs)):
                idx = int(idxs[i])
                src = dataset.paths[idx]
                rel = os.path.relpath(src, TEST_DIR)
                stem = os.path.splitext(rel)[0].replace(os.sep, "__")
                pred = preds[i].numpy().astype(np.uint8)

                # raw class-ID mask (same format as the training labels)
                Image.fromarray(pred).save(os.path.join(OUT_DIR, "masks", f"{stem}_pred.png"))

                # side-by-side: image | prediction | overlay
                if idx % SAVE_EVERY == 0:
                    img_np = images[i].permute(1, 2, 0).numpy()
                    color = colorize(pred)
                    overlay = (0.55 * img_np + 0.45 * color).astype(np.uint8)
                    panel = np.concatenate([img_np, color, overlay], axis=1)
                    Image.fromarray(panel).save(os.path.join(OUT_DIR, "vis", f"{stem}.png"))

            print(f"  processed {min(int(idxs[-1]) + 1, len(dataset))}/{len(dataset)}", end="\r")

    #  Class distribution of predictions 
    total = pixel_counts.sum().item()
    print("\n\nPredicted pixel share per class:")
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        share = pixel_counts[i].item() / total
        print(f"  {name:<14}{share * 100:>7.2f}%")
        rows.append({"class": name, "pixels": int(pixel_counts[i]), "share": share})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "pred_class_distribution.csv"), index=False)

    print(f"\nSaved to '{OUT_DIR}/': masks/ (class-ID PNGs), vis/ (image | prediction | overlay), distribution CSV.")


if __name__ == "__main__":
    main()