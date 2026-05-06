import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, list_file, target_size=(256, 256), is_test=False):
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_test = is_test
        
        # 依照助教建議的結構定義路徑 [cite: 171-174]
        self.img_folder = os.path.join(root_dir, 'images')
        self.mask_folder = os.path.join(root_dir, 'annotations', 'trimaps')
        
        # 讀取檔名清單 [cite: 11]
        with open(list_file, 'r') as f:
            self.file_ids = [line.strip().split()[0] for line in f]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # 1. 讀取與處理影像
        img_path = os.path.join(self.img_folder, f"{file_id}.jpg")
        image = Image.open(img_path).convert('RGB').resize(self.target_size)
        image = np.array(image).transpose(2, 0, 1) / 255.0  # (C, H, W) and normalize to [0, 1]
        
        if self.is_test:
            return torch.tensor(image, dtype=torch.float32), file_id
        
        # 2. 讀取與處理遮罩 (依照作業要求的二元化規則 [cite: 92-97])
        mask_path = os.path.join(self.mask_folder, f"{file_id}.png")
        mask = Image.open(mask_path).resize(self.target_size, resample=Image.NEAREST)
        mask_np = np.array(mask)
        
        # 規則：1->前景(1), 2&3->背景(0) [cite: 93-97]
        binary_mask = np.where(mask_np == 1, 1, 0).astype(np.float32)
        binary_mask = np.expand_dims(binary_mask, axis=0) # (1, H, W)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(binary_mask, dtype=torch.float32)
