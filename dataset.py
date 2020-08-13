import os
import torch
import numpy as np
from PIL import Image


class PennFudan(torch.utils.data.Dataset):
    
    def __init__(self, root, transforms=None, use_masks=False):
        self.root = root
        self.transforms = transforms
        self.images = list(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = list(os.listdir(os.path.join(root, "PedMasks")))
        self.use_masks = use_masks


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root, "PNGImages", self.images[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
    
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # remove bacground from masks
        object_ids = np.unique(mask)[1:]

        # Create boolean mask
        masks = mask == object_ids[:, None, None]

        num_objects = len(object_ids)
        boxes = list()

        for i in range(num_objects):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objects,), dtype=torch.int64)
        if self.use_masks:
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.as_tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        if self.use_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

