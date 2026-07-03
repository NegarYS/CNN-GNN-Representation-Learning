import torch
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
from collections import defaultdict
from typing import *

class SudokuMNIST(Dataset):
    def __init__(self, length=60000, train=True, root_dir='./data', seed=None):
        self.length = length
        self.train = train
        self.root_dir = root_dir
        self.seed = seed
        self.mnist = torchvision.datasets.MNIST(
            root=self.root_dir, train=self.train, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        self.digit_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.mnist):
            self.digit_indices[label].append(idx)
            
        self.min_sum = 0 + 1 + 2  # 3
        self.max_sum = 7 + 8 + 9  # 24
        self.sum_range = self.max_sum - self.min_sum

    def __len__(self):
        return self.length

    def _check_sort_order(self, arr):
        if arr[0] < arr[1] < arr[2]: return 1
        elif arr[0] > arr[1] > arr[2]: return 2
        else: return 0

    def __getitem__(self, idx):
        if self.seed is not None:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)
        all_digits = list(range(10))
        random.shuffle(all_digits)
        digits_for_grid = all_digits[:9]
        missing_digit_label = all_digits[9]
        random.shuffle(digits_for_grid)
        grid_digits = np.array(digits_for_grid).reshape(3, 3)
        grid_image = torch.zeros((1, 84, 84))
        for i in range(9):
            row, col = i // 3, i % 3
            digit_value = grid_digits[row, col]
            mnist_idx = random.choice(self.digit_indices[digit_value])
            digit_img, _ = self.mnist[mnist_idx]
            start_row, start_col = row * 28, col * 28
            grid_image[:, start_row:start_row + 28, start_col:start_col + 28] = digit_img

        row_sums = grid_digits.sum(axis=1)
        col_sums = grid_digits.sum(axis=0)
        sum_labels = np.concatenate([row_sums, col_sums])
        
        normalized_sum_labels = (sum_labels - self.min_sum) / self.sum_range

        row_sorts = [self._check_sort_order(grid_digits[i, :]) for i in range(3)]
        col_sorts = [self._check_sort_order(grid_digits[:, i]) for i in range(3)]
        sorted_labels = row_sorts + col_sorts

        output = {
            'image': grid_image,
            'missing_digit': torch.tensor(missing_digit_label, dtype=torch.long),
            'sorted_labels': torch.tensor(sorted_labels, dtype=torch.long),
            'sum_labels': torch.tensor(normalized_sum_labels, dtype=torch.float32)
        }
        
        return output

def get_datasets(root_dir) -> Tuple[Dataset, Dataset, Dataset]:
    train_ds = SudokuMNIST(length=50000, seed=42, root_dir=root_dir)
    val_ds = SudokuMNIST(length=10000, seed=43, root_dir=root_dir)
    test_ds = SudokuMNIST(length=20000, seed=44, root_dir=root_dir)

    return train_ds, val_ds, test_ds
    
if __name__ == "__main__":
    print("Initializing datasets...")
    train_ds, val_ds, test_ds = get_datasets("../data/")
    print("Sucessfully initialized all splits.\n")
    print(f"Length of train set: {len(train_ds)}")
    print(f"Length of validation set: {len(val_ds)}")
    print(f"Length of test set: {len(test_ds)}\n")