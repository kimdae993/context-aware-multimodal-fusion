from collections import Counter
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TwinMultiImageDataset(Dataset):
    """Session-level paired T1w/T2w slice dataset.

    ``img_group_list`` must contain one list per session. Each list should have
    ``num_slices`` T1w image paths followed by ``num_slices`` T2w image paths.
    """

    def __init__(
        self,
        img_group_list: List[List[str]],
        radscore_list: List[int],
        session_ids: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = "1vs5",
        num_slices: int = 3,
    ):
        if len(img_group_list) != len(radscore_list):
            raise ValueError("img_group_list and radscore_list must have the same length.")
        if mode not in {"1vs5", "1vs234vs5"}:
            raise ValueError("mode must be either '1vs5' or '1vs234vs5'.")

        self.num_slices = num_slices
        self.img_groups = img_group_list
        self.sessions = session_ids if session_ids is not None else [None] * len(self.img_groups)
        self.transform = transform if transform is not None else transforms.ToTensor()

        mapped = []
        if mode == "1vs5":
            for y in radscore_list:
                if y == 1:
                    mapped.append(0)
                elif y == 5:
                    mapped.append(1)
                else:
                    raise ValueError("1vs5 mode accepts only radiology scores 1 and 5.")
        else:
            for y in radscore_list:
                if y == 1:
                    mapped.append(0)
                elif y in [2, 3, 4]:
                    mapped.append(1)
                elif y == 5:
                    mapped.append(2)
                else:
                    raise ValueError(f"Unsupported radiology score: {y}")

        counts = Counter(mapped)
        if len(counts) < 2:
            raise ValueError("At least two classes must be present in the dataset.")

        expected_paths = 2 * self.num_slices
        for group in self.img_groups:
            if len(group) != expected_paths:
                raise ValueError(
                    f"Each sample must contain {expected_paths} paths "
                    f"({self.num_slices} T1w + {self.num_slices} T2w), but got {len(group)}."
                )

        self.label = torch.tensor(mapped, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def _load_grayscale(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        paths = self.img_groups[idx]
        t1_paths = paths[:self.num_slices]
        t2_paths = paths[self.num_slices:]

        t1_imgs = torch.stack([self._load_grayscale(p) for p in t1_paths], dim=0)
        t2_imgs = torch.stack([self._load_grayscale(p) for p in t2_paths], dim=0)

        return t1_imgs, t2_imgs, self.label[idx]
