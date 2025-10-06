# split_dataset.py
import os, random, shutil, argparse
from pathlib import Path

def split(images_dir, labels_dir, out_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    images = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in (".jpg",".png",".jpeg")]
    random.shuffle(images)
    ntrain = int(len(images)*train_ratio)
    train = images[:ntrain]
    val = images[ntrain:]

    img_train = Path(out_dir)/"images"/"train"
    img_val = Path(out_dir)/"images"/"val"
    lbl_train = Path(out_dir)/"labels"/"train"
    lbl_val = Path(out_dir)/"labels"/"val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_set(lst, img_dest, lbl_dest):
        for p in lst:
            shutil.copy(p, img_dest / p.name)
            lbl_src = Path(labels_dir)/ (p.stem + ".txt")
            if lbl_src.exists():
                shutil.copy(lbl_src, lbl_dest / lbl_src.name)
            else:
                # create empty label if not present
                open(lbl_dest / (p.stem + ".txt"), "w").close()

    copy_set(train, img_train, lbl_train)
    copy_set(val, img_val, lbl_val)
    print("Split done. Train:", len(train), "Val:", len(val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()
    split(args.images_dir, args.labels_dir, args.out_dir, args.train_ratio)
