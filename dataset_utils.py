# utils/dataset_utils.py
import os
import shutil
import random
from tqdm import tqdm

def split_dataset(src_root, dest_root, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    os.makedirs(dest_root, exist_ok=True)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_root, split), exist_ok=True)

    persons = sorted(os.listdir(src_root))
    for person in tqdm(persons, desc="Splitting identities"):
        src_dir = os.path.join(src_root, person)
        imgs = os.listdir(src_dir)
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        splits = {
            'train': imgs[:n_train],
            'val': imgs[n_train:n_train+n_val],
            'test': imgs[n_train+n_val:]
        }

        for split, files in splits.items():
            dst_dir = os.path.join(dest_root, split, person)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

if __name__ == "__main__":
    split_dataset(
        src_root="data/vggface2_subset_500",
        dest_root="data/vggface2_split"
    )
