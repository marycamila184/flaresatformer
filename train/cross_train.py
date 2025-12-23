import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.utils.binary_focal_loss import BinaryFocalLoss
from models.prithvi_model import PrithviSegmentation
from models.segformer_model import SegFormerB0
from models.unet_model import UNet

from train.utils.cross_split import create_folds
from train.utils.dataset import ImageMaskDataset
from train.utils.metrics import compute_metrics 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_FOLDS = 4
OUTPUT_DIR = "train/train_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

flare_patches = pd.read_csv("dataset/flare_dataset.csv")
urban_patches = pd.read_csv("dataset/urban_dataset.csv")
wildfire_patches = pd.read_csv("dataset/fire_dataset.csv")

images_flare, images_urban, images_wildfire = create_folds(
    flare_patches, urban_patches, wildfire_patches, NUM_FOLDS
)

list_models = ["unet", "segformer", "prithvi"]
dict_channels = [(1, 5, 6), (4, 5, 6), (3, 4, 5, 6), ()]

# Train
for model_name in list_models:
    for dict_bands in dict_channels:
        for fold in range(NUM_FOLDS):

            print(f"\n=== Model: {model_name} | Bands: {dict_bands} | Fold {fold+1} ===")

            flare_p = images_flare[images_flare["fold"] != fold]["tiff_file"]
            flare_m = images_flare[images_flare["fold"] != fold]["mask_file"]

            urban_p = images_urban[images_urban["fold"] != fold]["tiff_file"]
            urban_m = images_urban[images_urban["fold"] != fold]["mask_file"]

            fire_p = images_wildfire[images_wildfire["fold"] != fold]["tiff_file"]
            fire_m = images_wildfire[images_wildfire["fold"] != fold]["mask_file"]

            all_patches = pd.concat([flare_p, urban_p, fire_p]).tolist()
            all_masks = pd.concat([flare_m, urban_m, fire_m]).tolist()

            train_p, val_p, train_m, val_m = train_test_split(
                all_patches,
                all_masks,
                test_size=0.1,
                random_state=42,
                shuffle=True,
            )

            n_channels = len(dict_bands) if len(dict_bands) > 0 else 10

            train_ds = ImageMaskDataset(
                image_list=train_p,
                mask_list=train_m,
                n_channels=n_channels,
                bands=list(dict_bands),
                image_size=IMAGE_SIZE,
                target_resize=IMAGE_SIZE,
                augment=True,
            )

            val_ds = ImageMaskDataset(
                image_list=val_p,
                mask_list=val_m,
                n_channels=n_channels,
                bands=list(dict_bands),
                image_size=IMAGE_SIZE,
                target_resize=IMAGE_SIZE,
                augment=False,
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
            )

            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
            )

            # Model
            if model_name == "unet":
                model = UNet(in_channels=n_channels).to(DEVICE)

            elif model_name == "segformer":
                model = SegFormerB0(in_channels=n_channels).to(DEVICE)

            elif model_name == "prithvi":
                model = PrithviSegmentation(
                    in_channels=n_channels,
                    freeze_backbone=True
                ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = BinaryFocalLoss()

            best_f1 = 0.0
            history = []

            band_str = "".join(str(b) for b in dict_bands) if dict_bands else "all"
            ckpt_name = f"{model_name}_b{band_str}_fold{fold+1}.pth"
            ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)

            # Epochs
            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0.0

                for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Vaklidation
                model.eval()
                val_loss = 0.0
                all_preds, all_targets = [], []

                with torch.no_grad():
                    for imgs, masks in val_loader:
                        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                        logits = model(imgs)
                        loss = criterion(logits, masks)

                        val_loss += loss.item()
                        all_preds.append(torch.sigmoid(logits))
                        all_targets.append(masks)

                preds = torch.cat(all_preds)
                targets = torch.cat(all_targets)

                precision, recall, f1 = compute_metrics(preds, targets)

                history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss / len(train_loader),
                    "val_loss": val_loss / len(val_loader),
                    "val_precision": precision,
                    "val_recall": recall,
                    "val_f1": f1,
                })

                # Checkpoint
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), ckpt_path)

                print(
                    f"Epoch {epoch+1} | "
                    f"Val F1: {f1:.4f} | "
                    f"Precision: {precision:.4f} | "
                    f"Recall: {recall:.4f}"
                )

            # ---------
            # SAVE CSVs
            # ---------
            hist_df = pd.DataFrame(history)
            hist_df["fold"] = fold + 1

            model_dir = os.path.join(OUTPUT_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)

            hist_df.to_csv(
                os.path.join(model_dir, f"history_fold_{fold+1}_b{band_str}.csv"),
                index=False,
            )

            best_row = hist_df.loc[hist_df["val_f1"].idxmax()]
            summary = {
                "fold": fold + 1,
                "best_epoch": best_row["epoch"],
                "val_loss": best_row["val_loss"],
                "val_f1": best_row["val_f1"],
                "val_precision": best_row["val_precision"],
                "val_recall": best_row["val_recall"],
            }

            summary_df = pd.DataFrame([summary])
            summary_path = os.path.join(model_dir, f"summary_all_folds_b{band_str}.csv")
            summary_df.to_csv(
                summary_path,
                mode="a",
                header=not os.path.exists(summary_path),
                index=False,
            )
