import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import pandas as pd

NUM_CLASSES = 12
IGNORE_INDEX = 255
SUPER_CATEGORIES = [
    'vegetation', 'terrain', 'sky', 'construction',
    'vehicle', 'road', 'object', 'void', 'sign',
    'human', 'water', 'animal'
]
SUFFIXES = ["_camera_front", "_camera_left", "_camera_right", "_camera_back",
            "_front", "_left", "_right", "_back", "_realsense", "_color"]


# Dataset, returns compact uint8 tensors
class SmallGooseDataset(Dataset):
    def __init__(self, images_dir, labels_dir, size=(512, 256)):
        self.size = size
        self.items = []

        for split in os.listdir(images_dir):
            split_img = os.path.join(images_dir, split)
            split_lbl = os.path.join(labels_dir, split)
            if not os.path.isdir(split_img):
                continue

            for scenario in os.listdir(split_img):
                scen_img = os.path.join(split_img, scenario)
                scen_lbl = os.path.join(split_lbl, scenario)
                if not os.path.isdir(scen_lbl):
                    continue

                for img_name in os.listdir(scen_img):
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    base = os.path.splitext(img_name)[0]
                    for s in SUFFIXES:
                        if base.endswith(s):
                            base = base[:-len(s)]
                            break
                    mask_path = os.path.join(scen_lbl, f"{base}_labelids.png")
                    if os.path.exists(mask_path):
                        self.items.append((os.path.join(scen_img, img_name), mask_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        with Image.open(img_path) as im:
            im = im.convert("RGB").resize(self.size, Image.BILINEAR)
            img = np.asarray(im, dtype=np.uint8).copy()
        with Image.open(mask_path) as m:
            m = m.convert("L").resize(self.size, Image.NEAREST)
            mask = np.asarray(m, dtype=np.uint8).copy()

        # uint8 stays small; conversion to float/long happens on the GPU
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)


# Streaming confusion matrix, no need to keep all predictions
@torch.no_grad()
def update_confusion(cm, preds, targets, num_classes=NUM_CLASSES):
    valid = targets < num_classes
    idx = targets[valid] * num_classes + preds[valid]
    cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm):
    cm = cm.double()
    inter = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    class_iou = torch.where(union > 0, inter / union.clamp(min=1),
                            torch.full_like(inter, float('nan')))
    miou = torch.nanmean(class_iou).item()
    pixel_acc = (inter.sum() / cm.sum().clamp(min=1)).item()
    return miou, class_iou.cpu().tolist(), pixel_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    # Data: one dataset, split 80/20 (no more train == val)
    full = SmallGooseDataset("small-goose/images", "small-goose/labels12")
    n_val = max(1, int(0.2 * len(full)))
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kw = dict(batch_size=8, num_workers=2, pin_memory=use_amp,
                     persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

    def prep(images, masks):
        images = images.to(device, non_blocking=True).float()
        images = (images - mean) / std
        masks = masks.to(device, non_blocking=True).long()
        return images, masks


    model = smp.Linknet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_epochs = 10
    best_miou = -1.0
    history = []
    print("Starting LinkNet training...")

    for epoch in range(num_epochs):
        #Train
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = prep(images, masks)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        #  Validation, streaming metrics
        model.eval()
        val_loss = 0.0
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)

        with torch.inference_mode():
            for images, masks in val_loader:
                images, masks = prep(images, masks)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                update_confusion(cm, outputs.argmax(dim=1), masks)

        miou, class_iou, pixel_acc = metrics_from_confusion(cm)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "miou": miou,
            "pixel_acc": pixel_acc,
            **{f"iou_{n}": v for n, v in zip(SUPER_CATEGORIES, class_iou)},
        }
        history.append(row)
        pd.DataFrame(history).to_csv("metrics_history.csv", index=False) 

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {row['train_loss']:.4f} | "
              f"Val Loss: {row['val_loss']:.4f} | "
              f"mIoU: {miou:.4f} | Pixel Acc: {pixel_acc:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "linknet_best_model.pth")

    print(f"Done. Best mIoU: {best_miou:.4f}. Model and metrics saved.")


if __name__ == "__main__":
    main()