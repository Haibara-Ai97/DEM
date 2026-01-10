from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from models.dem_encoder import DEMEncoderConfig


CACHE_INDEX_COLUMNS = ("image_path", "cache_path")


class ImgCsv(Dataset):
    def __init__(self, csv_path: str):
        import pandas as pd

        df = pd.read_csv(csv_path)
        self.paths = df["image_path"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return path, img


class CacheIndexDataset(Dataset):
    def __init__(self, index_csv: str):
        import pandas as pd

        df = pd.read_csv(index_csv)
        if not set(CACHE_INDEX_COLUMNS).issubset(df.columns):
            raise ValueError(f"Cache index must contain columns: {CACHE_INDEX_COLUMNS}")
        self.image_paths = df["image_path"].tolist()
        self.cache_paths = df["cache_path"].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image, str]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.image_paths[idx], img, self.cache_paths[idx]


def pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def normalize(t: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


class CacheCollator:
    def __init__(self, enc_cfg: "DEMEncoderConfig", image_size: int):
        self.enc_cfg = enc_cfg
        self.image_size = image_size

    def __call__(self, batch: list[tuple[str, Image.Image, str]]):
        paths, imgs, cache_paths = zip(*batch)

        enc_imgs = [pil_resize_square(im, self.image_size) for im in imgs]
        enc_x = torch.stack(
            [normalize(pil_to_tensor(im), self.enc_cfg.image_mean, self.enc_cfg.image_std) for im in enc_imgs], dim=0
        )

        topi_list, topv_list, hw_list = [], [], []
        for cp in cache_paths:
            z = np.load(cp)
            topi_list.append(torch.from_numpy(z["topi"]).long())
            topv_list.append(torch.from_numpy(z["topv"]).float())
            h = int(z["h"])
            w = int(z["w"])
            hw_list.append((h, w))

        if len(set(hw_list)) != 1:
            raise ValueError(f"Cache grid differs inside a batch: {hw_list}")
        h, w = hw_list[0]

        topi = torch.stack(topi_list, dim=0)
        topv = torch.stack(topv_list, dim=0)

        return list(paths), enc_x, topi, topv, (h, w)
