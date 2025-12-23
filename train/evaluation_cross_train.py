import os
from models.prithvi_model import PrithviSegmentation
from models.segformer_model import SegFormerB0
from models.unet_model import UNet
import torch

import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader

from train.utils.cross_split import create_folds
from train.utils.dataset import ImageMaskDataset
from train.utils.metrics import compute_metrics


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.5
NUM_FOLDS = 4
OUTPUT_DIR = "train/train_output"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8

model_name_map = {
    "unet": "UNet",
    "segformer": "SegFormer-B0",
    "prithvi": "Prithvi",
}

channels_readable = {
    "156": "2, 6 e 7",
    "456": "5, 6 e 7",
    "3456": "4, 5, 6 e 7",
    "10": "1 a 10",
}

list_models = ["unet", "segformer", "prithvi"]
list_bands = [[1, 5, 6], [4, 5, 6], [3, 4, 5, 6], []]
path_channels = ["156", "456", "3456", "10"]


flare_patches = pd.read_csv("dataset/flare_dataset.csv")
urban_patches = pd.read_csv("dataset/urban_dataset.csv")
wildfire_patches = pd.read_csv("dataset/fire_dataset.csv")

images_flare, images_urban, images_wildfire = create_folds(
    flare_patches, urban_patches, wildfire_patches, NUM_FOLDS
)

results = []

# Evaluation
for model_name in list_models:
    pretty_name = model_name_map[model_name]

    for idx, bands in enumerate(list_bands):
        n_channels = len(bands) if len(bands) > 0 else 10

        row = {
            "Model": pretty_name,
            "Channels": n_channels,
            "Bands": channels_readable[path_channels[idx]],
        }

        for fold in range(NUM_FOLDS):
            print(f"Evaluating {model_name} | Bands {bands} | Fold {fold+1}")

            # Model
            if model_name == "unet":
                model = UNet(in_channels=n_channels)

            elif model_name == "segformer":
                model = SegFormerB0(in_channels=n_channels)

            elif model_name == "prithvi":
                model = PrithviSegmentation(in_channels=n_channels)

            model.to(DEVICE)

            ckpt_name = f"{model_name}_b{path_channels[idx]}_fold{fold+1}.pth"
            ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.eval()

            # Test Split
            flare_p = images_flare[images_flare["fold"] == fold]["tiff_file"]
            flare_m = images_flare[images_flare["fold"] == fold]["mask_file"]

            urban_p = images_urban[images_urban["fold"] == fold]["tiff_file"]
            urban_m = images_urban[images_urban["fold"] == fold]["mask_file"]

            fire_p = images_wildfire[images_wildfire["fold"] == fold]["tiff_file"]
            fire_m = images_wildfire[images_wildfire["fold"] == fold]["mask_file"]

            all_patches = pd.concat([flare_p, urban_p, fire_p]).tolist()
            all_masks = pd.concat([flare_m, urban_m, fire_m]).tolist()

            test_ds = ImageMaskDataset(
                image_list=all_patches,
                mask_list=all_masks,
                n_channels=n_channels,
                bands=bands,
                image_size=IMAGE_SIZE,
                target_resize=IMAGE_SIZE,
                augment=False,
            )

            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
            )

            # Inference
            all_preds, all_targets = [], []

            with torch.no_grad():
                for imgs, masks in tqdm(test_loader):
                    imgs = imgs.to(DEVICE)
                    masks = masks.to(DEVICE)

                    logits = model(imgs)
                    probs = torch.sigmoid(logits)

                    all_preds.append(probs)
                    all_targets.append(masks)

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)

            precision, recall, f1 = compute_metrics(
                preds, targets, threshold=THRESHOLD
            )

            # IoU
            preds_bin = (preds > THRESHOLD).float()
            intersection = (preds_bin * targets).sum().item()
            union = ((preds_bin + targets) > 0).sum().item()
            iou = intersection / (union + 1e-7)

            suffix = f"({fold+1})"
            row[f"F1 - {suffix}"] = round(f1, 4)
            row[f"P - {suffix}"] = round(precision, 4)
            row[f"R - {suffix}"] = round(recall, 4)
            row[f"IoU - {suffix}"] = round(iou, 4)

        results.append(row)

# Save Results
df = pd.DataFrame(results)

columns = ["Model", "Channels", "Bands"]
for i in range(1, NUM_FOLDS + 1):
    columns += [
        f"F1 - ({i})",
        f"P - ({i})",
        f"R - ({i})",
        f"IoU - ({i})",
    ]

df = df[columns]

evaluation_path = os.path.join(OUTPUT_DIR, "cross_train_results.csv")
df.to_csv(evaluation_path, index=False)
