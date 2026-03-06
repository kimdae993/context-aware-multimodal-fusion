from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter


class TwinMultiImageDataset(Dataset):
    """
    Dataset for paired multi-slice T1/T2 image inputs.

    Each sample is expected to contain six image paths in the following order:
        [T1_slice1, T1_slice2, T1_slice3, T2_slice1, T2_slice2, T2_slice3]

    Radiology scores are mapped to binary class labels as:
        - radscore 1 -> class 0
        - radscore 5 -> class 1

    Args:
        img_group_list:
            List of image-path groups. Each group must contain 6 paths.
        radscore_list:
            List of radiology scores. Expected values are 1 and 5.
        session_ids:
            Optional session identifiers aligned with `img_group_list`.
        transform:
            Transform applied to each grayscale PIL image.
            If None, `transforms.ToTensor()` is used.

    Returns from `__getitem__`:
        t1_imgs: Tensor of shape [3, 1, H, W]
        t2_imgs: Tensor of shape [3, 1, H, W]
        y: Tensor scalar label
        paths: Original list of 6 image paths
    """

    def __init__(
        self,
        img_group_list: List[List[str]],
        radscore_list: List[int],
        session_ids: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        if len(img_group_list) != len(radscore_list):
            raise ValueError(
                "The number of image groups must match the number of labels. "
                f"Got {len(img_group_list)} image groups and {len(radscore_list)} labels."
            )

        if session_ids is not None and len(session_ids) != len(img_group_list):
            raise ValueError(
                "The number of session IDs must match the number of image groups. "
                f"Got {len(session_ids)} session IDs and {len(img_group_list)} image groups."
            )

        for index, paths in enumerate(img_group_list):
            if len(paths) != 6:
                raise ValueError(
                    f"Each sample must contain exactly 6 image paths "
                    f"(3 T1 + 3 T2). Sample index {index} has {len(paths)} paths."
                )

        self.img_groups = img_group_list

        try:
            mapped_labels = [0 if int(score) == 1 else 1 for score in radscore_list]
        except Exception as exc:
            raise ValueError(
                "Failed to parse `radscore_list`. Expected values that can be interpreted as 1 or 5."
            ) from exc

        unique_scores = {int(score) for score in radscore_list}
        if not unique_scores.issubset({1, 5}):
            raise ValueError(
                f"`radscore_list` must contain only 1 and 5. Got: {sorted(unique_scores)}"
            )

        counts = Counter(mapped_labels)
        if len(counts.keys()) != 2:
            raise ValueError(
                "Both classes must be present in the dataset after mapping. "
                f"Observed class counts: {dict(counts)}"
            )

        self.labels = torch.tensor(mapped_labels, dtype=torch.long)
        self.sessions = session_ids if session_ids is not None else [None] * len(self.img_groups)
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def _load_grayscale_image(self, path: str) -> torch.Tensor:
        """
        Load a grayscale image and apply the configured transform.

        Args:
            path: Path to the image file.

        Returns:
            Tensor of shape [1, H, W] after transformation.
        """
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("L")
            tensor = self.transform(image) if self.transform is not None else image

        return tensor

    def __getitem__(self, idx: int):
        """
        Load one paired T1/T2 sample.

        Args:
            idx: Sample index.

        Returns:
            t1_imgs:
                Tensor of shape [3, 1, H, W]
            t2_imgs:
                Tensor of shape [3, 1, H, W]
            y:
                Label tensor
            paths:
                Original list of file paths
        """
        paths = self.img_groups[idx]

        # First 3 paths: T1 slices, last 3 paths: T2 slices
        t1_paths = paths[:3]
        t2_paths = paths[3:]

        t1_imgs = [self._load_grayscale_image(path) for path in t1_paths]
        t2_imgs = [self._load_grayscale_image(path) for path in t2_paths]

        t1_imgs = torch.stack(t1_imgs, dim=0)  # [3, 1, H, W]
        t2_imgs = torch.stack(t2_imgs, dim=0)  # [3, 1, H, W]

        y = self.labels[idx]
        return t1_imgs, t2_imgs, y