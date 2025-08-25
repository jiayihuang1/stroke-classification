import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class StrokeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """ Defines a subclass of torch.utils.data.Dataset, which is the required interface for PyTorch DataLoader.

        Args:
            image_paths (list): Paths to CT images.
            labels (list): Corresponding labels (0 for stroke absent, 1 for stroke present).
            transform (callable, optional): Data augmentation and normalization pipeline.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Fetches the image at specified index, processes image and applies transformation (if any) and returns the image and its corresponding label in tensor format

        Args:
            idx (int): index of the desired image

        Returns:
            image (tensor): transformed image
            label (tensor): corresponding label
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label