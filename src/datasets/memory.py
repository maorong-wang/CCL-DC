from __future__ import print_function
from torch.utils.data import Dataset

class MemoryDataset(Dataset):
    """MemoryDataset for loading memory data.
    """
    def __init__(self, tensor_imgs, tensor_labels, transform=None):
        super().__init__()
        self.data = tensor_imgs
        self.targets = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, target
