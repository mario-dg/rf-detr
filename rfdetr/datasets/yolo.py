"""
YOLO dataset loader.
Optimized for large datasets to avoid the memory overhead of converting beforehand.
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.data
from PIL import Image
from supervision.utils.file import (
    list_files_with_extensions,
    read_txt_file,
    read_yaml_file,
)

from rfdetr.datasets.coco import (
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)


class YOLODataset(torch.utils.data.Dataset):
    """Dataset for YOLO format annotations"""

    DEFAULT_YOLO_YAML_FILENAME = "data.yaml"
    DEFAULT_SPLIT_DIRS = ["train", "valid"]
    REQUIRED_DATA_SUBDIRS = ["images", "labels"]

    def __init__(
        self,
        dataset_directory_path: Union[str, Path],
        transforms: Callable,
        dataset_split_names: Optional[Union[str, List[str]]] = None,
        data_yaml_filename: Optional[str] = None,
        run_validation: bool = True,
    ):
        """
        Initializes the YOLODataset object by loading dataset structure, class names and image-label pairs.

        Args:
            dataset_directory_path (Union[str, Path]): Path to the root directory of the YOLO dataset.
            transforms (Callable): Transformations to apply to the dataset samples.
            dataset_split_names (Optional[Union[str, List[str]]], optional): Name(s) of dataset splits (e.g., 'train', 'valid', 'test').
            data_yaml_filename (Optional[str], optional): Filename of the YOLO data YAML file.
            run_validation (bool, optional): Whether to validate the dataset structure upon initialization.

        Side Effects:
            - Loads class names from the data YAML file.
            - Collects and matches image and label file paths for each split.
            - Prints information about loaded classes and found image-label pairs.
        """
        super(YOLODataset, self).__init__()

        # Initialize dataset parameters
        if isinstance(dataset_directory_path, str):
            dataset_directory_path = Path(dataset_directory_path)
        if dataset_split_names is None:
            dataset_split_names = self.DEFAULT_SPLIT_DIRS
        if isinstance(dataset_split_names, str):
            dataset_split_names = [dataset_split_names]
        dataset_split_names = list(set(dataset_split_names))
        if data_yaml_filename is None:
            data_yaml_filename = self.DEFAULT_YOLO_YAML_FILENAME

        # Validate the dataset structure
        if run_validation:
            self.is_valid_dataset(
                dataset_directory_path=dataset_directory_path,
                data_yaml_filename=data_yaml_filename,
                split_dirs=dataset_split_names,
            )

        # Load class names from the data.yaml file
        data_yaml_path = dataset_directory_path / data_yaml_filename
        data = read_yaml_file(data_yaml_path)
        self.class_names = data.get("names", [])
        print(f"Loaded {len(self.class_names)} classes: {self.class_names}")

        # Initialize image and label paths
        self.image_paths, self.label_paths = [], []
        for split_name in dataset_split_names:
            images_directory_path = dataset_directory_path / split_name / "images"
            image_paths = list_files_with_extensions(
                directory=images_directory_path,
                extensions=["jpg", "jpeg", "png"],
            )

            labels_directory_path = dataset_directory_path / split_name / "labels"
            label_paths = list_files_with_extensions(
                directory=labels_directory_path,
                extensions=["txt"],
            )

            image_paths, label_paths = self._match_image_label_pairs(
                image_paths=image_paths, label_paths=label_paths
            )

            self.image_paths.extend(image_paths)
            self.label_paths.extend(label_paths)

        print(f"Found {len(self.image_paths)} valid image-label pairs.")
        self.ids = list(range(1, len(self.image_paths) + 1))
        self.transforms = transforms
        self._coco = None

    @staticmethod
    def is_valid_dataset(
        dataset_directory_path: Union[str, Path],
        data_yaml_filename: str,
        split_dirs: Optional[List[str]] = None,
    ):
        """
        Validate that the specified directory follows the expected YOLO dataset structure.

        Requirements:
        - The root directory must contain a data.yaml file.
        - Each split directory (e.g., 'train', 'valid') must exist and contain both 'images' and 'labels' subdirectories.
        - The 'test' split is optional and not required for validation.

        Args:
            dataset_directory_path (Union[str, Path]): Path to the root of the dataset.
            data_yaml_filename (str): Name of the YAML file describing the dataset.
            split_dirs (Optional[List[str]]): List of split directories to check (e.g., ['train', 'valid']).

        Returns:
            None. Prints warnings if the structure is invalid.
        """
        if isinstance(dataset_directory_path, str):
            dataset_directory_path = Path(dataset_directory_path)

        data_yaml_path = dataset_directory_path / data_yaml_filename
        if not data_yaml_path.exists():
            print(f"Missing {data_yaml_filename} in {dataset_directory_path}")

        if split_dirs is None:
            split_dirs = YOLODataset.DEFAULT_SPLIT_DIRS

        for split_dir in split_dirs:
            split_dir_path = dataset_directory_path / split_dir
            if not split_dir_path.exists():
                print(f"Missing {split_dir} directory in {dataset_directory_path}")

            for data_subdir in YOLODataset.REQUIRED_DATA_SUBDIRS:
                if not (split_dir_path / data_subdir).exists():
                    print(f"Missing {data_subdir} directory in {split_dir_path}")

    @staticmethod
    def _match_image_label_pairs(
        image_paths: List[Path],
        label_paths: List[Path],
    ) -> Tuple[List[Path], List[Path]]:
        """
        Matches image paths with their corresponding label paths.

        Args:
            image_paths: List of paths to image files
            label_paths: List of paths to label files

        Returns:
            Tuple of (matched_image_paths, matched_label_paths) with paired files in sorted order
        """

        # Check if the image list is empty
        len_image_paths = len(image_paths)
        if len_image_paths == 0:
            print("No images found, returning empty lists.")
            return [], []

        # Check if the label list is empty
        len_label_paths = len(label_paths)
        if len_label_paths == 0:
            print("No labels found, returning empty lists.")
            return [], []

        # Check if the number of images and labels match
        if len_image_paths != len_label_paths:
            print(
                f"Warning: Found {len_image_paths} images and {len_label_paths} labels. "
                + "Matching will be performed based on file names."
            )

        # Determine equal stems between images and labels
        image_stems = {path.stem: path for path in image_paths}
        label_stems = {path.stem: path for path in label_paths}
        common_stems = set(image_stems.keys()) & set(label_stems.keys())
        len_common_stems = len(common_stems)

        # Check if there are more images than labels
        if len_common_stems != len_image_paths:
            unmatched_image_stems = set(image_stems.keys()) - common_stems
            print(
                f"Warning: Found {len(unmatched_image_stems)} images without matching labels"
            )
            if len(unmatched_image_stems) <= 10:
                print(f"  Unmatched images: {', '.join(unmatched_image_stems)}")
            else:
                print(
                    f"  First 10 unmatched images: {', '.join(list(unmatched_image_stems)[:10])}..."
                )

        # Check if there are more labels than images
        if len_common_stems != len_label_paths:
            unmatched_label_stems = set(label_stems.keys()) - common_stems
            print(
                f"Warning: Found {len(unmatched_label_stems)} labels without matching images"
            )
            if len(unmatched_label_stems) <= 10:
                print(f"  Unmatched labels: {', '.join(unmatched_label_stems)}")
            else:
                print(
                    f"  First 10 unmatched labels: {', '.join(list(unmatched_label_stems)[:10])}..."
                )

        # Sort the common stems and create matched paths
        sorted_common_stems = sorted(common_stems)
        matched_image_paths = [image_stems[stem] for stem in sorted_common_stems]
        matched_label_paths = [label_stems[stem] for stem in sorted_common_stems]

        return matched_image_paths, matched_label_paths

    @staticmethod
    def _parse_yolo_annotations(
        lines: List[str],
        resolution_wh: Tuple[int, int],
        class_names: List[str],
    ) -> Tuple[List[int], List[float]]:
        """
        Parse YOLO annotation lines into class labels and bounding box coordinates.

        Each line must be in the format:
            <class_id> <x_center> <y_center> <width> <height>
        where all coordinates are normalized to [0, 1].

        Args:
            lines (List[str]): Annotation lines in YOLO format.
            resolution_wh (Tuple[int, int]): (width, height) of the image.
            class_names (List[str]): List of class names.

        Returns:
            Tuple[List[int], List[List[float]]]:
                - List of valid class IDs.
                - List of bounding boxes as [x1, y1, x2, y2] in pixel coordinates.

        Notes:
            - Lines with invalid format, class IDs, or coordinates are skipped with a warning.
        """
        len_class_names = len(class_names)

        labels, boxes = [], []
        for line in lines:
            line_parts = line.strip().split()

            # Skip lines with invalid format:
            if len(line_parts) != 5:
                print(f"Warning: Skipping invalid line format: {line.strip()}")
                continue

            # Check class ID:
            class_id = int(line_parts[0])
            if class_id < 0 or class_id >= len_class_names:
                print(f"Warning: Skipping invalid class ID {class_id}")
                continue

            # Check coordinates:
            x_center, y_center, width, height = map(float, line_parts[1:5])
            if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                print(
                    f"Warning: Skipping invalid coordinates {x_center}, {y_center}, {width}, {height}. (Not normalized)"
                )
                continue

            labels.append(class_id)

            x1 = (x_center - width / 2) * resolution_wh[0]
            y1 = (y_center - height / 2) * resolution_wh[1]
            x2 = (x_center + width / 2) * resolution_wh[0]
            y2 = (y_center + height / 2) * resolution_wh[1]
            boxes.append([x1, y1, x2, y2])

        return labels, boxes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = int(idx)
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image_id = self.ids[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        target = {}
        target["image_id"] = torch.tensor([image_id])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        label_lines = read_txt_file(label_path)
        labels, boxes = self._parse_yolo_annotations(
            label_lines, (w, h), self.class_names
        )

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if len(boxes) > 0
            else torch.zeros(0)
        )
        target["iscrowd"] = torch.zeros_like(labels, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @property
    def coco(self):
        """
        Return a COCO-like API object for compatibility with pycocotools evaluation
        """
        if self._coco is None:
            self._coco = CocoLikeAPI(self)
        return self._coco


class CocoLikeAPI:
    """
    A COCO-like API for compatibility with pycocotools evaluation.
    This simulates the COCO API used for evaluation.
    """

    def __init__(self, dataset: YOLODataset):
        self.orig_dataset = dataset
        self.cats = self._create_category_mapping()
        self.imgs = self._create_image_mapping()
        self.anns = self._create_annotation_mapping()

        self.imgToAnns = defaultdict(list)
        self.catToImgs = defaultdict(list)

        for ann in self.anns.values():
            self.imgToAnns[ann["image_id"]].append(ann)
            self.catToImgs[ann["category_id"]].append(ann["image_id"])

        self.dataset = {
            "images": self.imgs,
            "annotations": list(self.anns.values()),
            "categories": list(self.cats.values()),
        }

    def _create_category_mapping(self):
        """Create a category mapping similar to COCO format"""
        cats = {}
        for idx, name in enumerate(self.orig_dataset.class_names):
            cat_id = idx
            cats[cat_id] = {"id": cat_id, "name": name, "supercategory": "none"}
        return cats

    def _create_image_mapping(self):
        """Create an image mapping similar to COCO format"""
        imgs = []
        for idx, img_path in enumerate(self.orig_dataset.image_paths):
            img = Image.open(img_path)
            width, height = img.size
            imgs.append(
                {
                    "id": self.orig_dataset.ids[idx],
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                }
            )
        return imgs

    def _create_annotation_mapping(self):
        """Create an annotation mapping similar to COCO format"""
        anns = {}
        ann_id = 0

        for idx, (img_path, label_path) in enumerate(
            zip(self.orig_dataset.image_paths, self.orig_dataset.label_paths)
        ):
            img = Image.open(img_path)
            width, height = img.size

            with open(label_path, "r") as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x_center, y_center, box_width, box_height = map(
                            float, data[1:5]
                        )

                        x = (x_center - box_width / 2) * width
                        y = (y_center - box_height / 2) * height
                        w = box_width * width
                        h = box_height * height

                        anns[ann_id] = {
                            "id": ann_id,
                            "image_id": self.orig_dataset.ids[idx],
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                        ann_id += 1

        return anns

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """Get annotation IDs matching the given filter conditions"""
        anns = self.anns.values()

        if imgIds is not None:
            if not isinstance(imgIds, list):
                imgIds = [imgIds]
            anns = [ann for ann in anns if ann["image_id"] in imgIds]

        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]
            anns = [ann for ann in anns if ann["category_id"] in catIds]

        if areaRng is not None:
            anns = [ann for ann in anns if areaRng[0] <= ann["area"] <= areaRng[1]]

        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]

        return [ann["id"] for ann in anns]

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """Get category IDs matching the given filter conditions"""
        cats = self.cats.values()

        if catNms is not None:
            if not isinstance(catNms, list):
                catNms = [catNms]
            cats = [cat for cat in cats if cat["name"] in catNms]

        if supNms is not None:
            if not isinstance(supNms, list):
                supNms = [supNms]
            cats = [cat for cat in cats if cat["supercategory"] in supNms]

        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]
            cats = [cat for cat in cats if cat["id"] in catIds]

        return [cat["id"] for cat in cats]

    def getImgIds(self, imgIds=None, catIds=None):
        """Get image IDs matching the given filter conditions"""
        imgs = self.imgs

        if imgIds is not None:
            if not isinstance(imgIds, list):
                imgIds = [imgIds]
            imgs = [img for img in imgs if img["id"] in imgIds]

        if catIds is not None:
            if not isinstance(catIds, list):
                catIds = [catIds]

            img_ids = set()
            for cat_id in catIds:
                img_ids.update(self.catToImgs[cat_id])
            imgs = [img for img in imgs if img["id"] in img_ids]

        return [img["id"] for img in imgs]

    def loadAnns(self, ids):
        """Load annotations with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.anns[id] for id in ids if id in self.anns]

    def loadCats(self, ids):
        """Load categories with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.cats[id] for id in ids if id in self.cats]

    def loadImgs(self, ids):
        """Load images with the specified IDs"""
        if isinstance(ids, int):
            ids = [ids]
        return [self.imgs[id] for id in ids if id in self.imgs]


def build_yolo(
    image_set: str,
    args: argparse.Namespace,
    resolution: int,
) -> YOLODataset:
    """
    Builds and returns a YOLODataset instance for the specified image set.

    Args:
        image_set (str): The dataset split to use. Must be one of "train", "val", or "test".
        args (argparse.Namespace): Parsed command-line arguments containing dataset configuration.
        resolution (int): Target image resolution for transformations.
    Returns:
        YOLODataset: Configured dataset instance for the specified split.
    Raises:
        KeyError: If `image_set` is not one of the expected values.
    """
    dataset_directory_path = Path(args.dataset_dir)

    dataset_split_name = image_set.split("_")[0]
    if dataset_split_name == "val":
        dataset_split_name = "valid"  # Handle common YOLO split name

    try:
        square_resize_div_64 = args.square_resize_div_64
    except AttributeError:  # `square_resize_div_64` not set in args
        transforms_sink = make_coco_transforms
    else:
        transforms_sink = make_coco_transforms_square_div_64

    try:
        multi_scale = args.multi_scale
    except AttributeError:
        multi_scale = False

    try:
        expanded_scales = args.expanded_scales
    except AttributeError:
        expanded_scales = False

    return YOLODataset(
        dataset_directory_path=dataset_directory_path,
        dataset_split_names=dataset_split_name,
        transforms=transforms_sink(
            image_set,
            resolution,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
        ),
    )


def _dev():
    """
    Development function to test the YOLODataset class.
    """
    from os import environ
    
    dataset_dir = environ.get('COCO2017_YOLO_DIR')
    dataset = build_yolo(
        image_set="train",
        args=argparse.Namespace(
            dataset_dir=dataset_dir,
        ),
        resolution=448,
    )
    print(f"Loaded {len(dataset)} samples from the dataset.")


if __name__ == "__main__":
    _dev()
