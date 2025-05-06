import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_dataset(
    images_dir,
    labels_dir,
    output_dir,
    train_ratio=0.7,
    seed=42
):
    random.seed(seed)

    # Convert to Path objects
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Get all image files (assuming .jpg or .png)
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"Total images found: {len(image_files)}")

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Output folders
    train_img_dir = output_dir / "train" / "images" 
    val_img_dir   = output_dir / "val" / "images" 
    train_lbl_dir = output_dir / "train" / "labels" 
    val_lbl_dir   = output_dir / "val" / "labels" 

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    for subset, img_list, img_out, lbl_out in [
        ("train", train_files, train_img_dir, train_lbl_dir),
        ("val", val_files, val_img_dir, val_lbl_dir)
    ]:
        print(f"Copying {subset} data...")
        for img_path in tqdm(img_list):
            # Copy image
            shutil.copy(img_path, img_out / img_path.name)

            # Copy corresponding label
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, lbl_out / label_path.name)
            else:
                print(f"Warning: Label not found for {img_path.name}")

    print("Dataset split completed!")

if __name__ == "__main__":
    split_dataset(
        # change the path of the img, and labels dir based on your img, and labels path
        images_dir="/home/gautamarora/project/dark_face/val/images",
        labels_dir="/home/gautamarora/project/dark_face/val/labels",
        output_dir="/home/gautamarora/project/dark_face/split"
    )
