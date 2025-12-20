import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class PlantDataset(Dataset):
    """Dataset class that reads images from a folder and applies transformations."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            class_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_folder):
                self.samples.append(
                    (os.path.join(class_folder, img_name), self.class_to_idx[cls])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



# DATA PIPELINE CLASS 
class DataPipeline:
    """Handles all preprocessing, transformations, and DataLoaders."""
    def __init__(self, data_dir, batch_size=32, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        self.train_transform, self.test_transform = self._get_transforms()

    def _get_transforms(self):
        train_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

        test_tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        return train_tf, test_tf

    def get_dataloaders(self):
        """Return train, val, test DataLoaders."""
        train_ds = PlantDataset(
            os.path.join(self.data_dir, "train"),
            transform=self.train_transform
        )
        val_ds = PlantDataset(
            os.path.join(self.data_dir, "val"),
            transform=self.test_transform
        )
        test_ds = PlantDataset(
            os.path.join(self.data_dir, "test"),
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader

    def get_class_names(self):
        """Return list of class names in the dataset."""
        train_ds = PlantDataset(os.path.join(self.data_dir, "train"))
        return train_ds.classes
        
# TESTING
if __name__ == "__main__":
    pipeline = DataPipeline(data_dir="data/processed", batch_size=32, img_size=224)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Classes:", pipeline.get_class_names())
