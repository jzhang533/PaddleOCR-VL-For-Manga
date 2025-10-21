import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset


load_dotenv()

MANGA109_ROOT = Path(os.getenv("MANGA109_ROOT", "")).expanduser()
DATA_SYNTHETIC_ROOT = Path(os.getenv("DATA_SYNTHETIC_ROOT", "")).expanduser()

CHOSEN_TASK = "ocr"
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}


class MangaDataset(Dataset):
    """
    Dataset that returns raw image-text pairs.
    Preprocessing is handled by the collator.
    """

    def __init__(
        self,
        split,
        limit_size=None,
        augment=False,
        skip_packages=None,
        use_synthetic=True,
    ):
        data = []

        print(f"Initializing dataset {split}...")

        if skip_packages is None:
            skip_packages = set()
        else:
            skip_packages = {f"{x:04d}" for x in skip_packages}

        # Load synthetic data only if use_synthetic=True
        if use_synthetic:
            for path in sorted((DATA_SYNTHETIC_ROOT / "meta").glob("*.csv")):
                if path.stem in skip_packages:
                    print(f"Skipping package {path}")
                    continue
                if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                    print(f"Missing image data for package {path}, skipping")
                    continue
                df = pd.read_csv(path)
                df = df.dropna()
                df["path"] = df.id.apply(
                    lambda x, path_stem=path.stem: str(
                        DATA_SYNTHETIC_ROOT / "img" / path_stem / f"{x}.jpg"
                    )
                )
                df = df[["path", "text"]]
                df["synthetic"] = True
                data.append(df)

        # Load MANGA109 data - use split filter only for non-synthetic data
        df = pd.read_csv(MANGA109_ROOT / "data.csv")
        df = df[df.split == split].reset_index(drop=True)
        df["path"] = df.crop_path.apply(lambda x: str(MANGA109_ROOT / x))
        df = df[["path", "text"]]
        df["synthetic"] = False
        data.append(df)

        data = pd.concat(data, ignore_index=True)

        if limit_size:
            data = data.iloc[:limit_size]
        self.data = data

        print(f"Dataset {split}: {len(self.data)}")

        self.augment = augment
        self.transform_medium, self.transform_heavy = self.get_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return image and messages in the format expected by VL models.

        Returns:
            dict: {
                'images': List[PIL.Image.Image] - List containing RGB image
                'messages': List[dict] - List of message dicts with role and content
            }
        """
        sample = self.data.loc[idx]
        text = sample.text

        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(
                ["none", "medium", "heavy"],
                p=[1 - medium_p - heavy_p, medium_p, heavy_p],
            )
            transform = {
                "none": None,
                "medium": self.transform_medium,
                "heavy": self.transform_heavy,
            }[transform_variant]
        else:
            transform = None

        image = self.read_image(sample.path, transform)

        return {
            "images": [image],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }

    @staticmethod
    def read_image(path, transform=None):  # noqa: ARG004
        """Read and transform image from path."""
        image = Image.open(path).convert("RGB")
        return image

    @staticmethod
    def get_transforms():
        t_medium = A.Compose(
            [
                A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.InvertImg(p=0.05),
                A.OneOf(
                    [
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
                    ],
                    p=0.1,
                ),
                A.Blur(p=0.2),
                A.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise((50, 200), p=0.3),
                A.ImageCompression(1, 30, p=0.1),
                A.ToGray(always_apply=True),
            ]
        )

        t_heavy = A.Compose(
            [
                A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.InvertImg(p=0.05),
                A.OneOf(
                    [
                        A.Downscale(0.1, 0.2, interpolation=cv2.INTER_LINEAR),
                        A.Downscale(0.1, 0.2, interpolation=cv2.INTER_NEAREST),
                    ],
                    p=0.1,
                ),
                A.Blur((3, 9), p=0.5),
                A.Sharpen(p=0.5),
                A.RandomBrightnessContrast(0.8, 0.8, p=1),
                A.GaussNoise((1000, 10000), p=0.3),
                A.ImageCompression(1, 10, p=0.5),
                A.ToGray(always_apply=True),
            ]
        )

        return t_medium, t_heavy


if __name__ == "__main__":
    ds = MangaDataset("train", augment=True)

    for i in range(5):
        sample = ds[i]
        print(f"{i}:")
        print(f"  Images: {len(sample['images'])} image(s)")
        print(f"  Image size: {sample['images'][0].size}")
        print(f"  Messages: {sample['messages']}")
        print()
