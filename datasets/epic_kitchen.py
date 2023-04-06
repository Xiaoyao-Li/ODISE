import time
import os
import json
import pickle
from typing import Tuple

import numpy as np
import torch

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.detection_utils import read_image
from torch.utils.data import Dataset, DataLoader

from datasets.misc import collate_fn_general

from detectron2.config import LazyConfig, instantiate
from icecream import ic

class EpicKitchen(Dataset):
    """ Dataset for epic kitchen
    """
    LENGTH_FRAME_ID = 10
    def __init__(self, part: str, clip: str, aug: T.AugmentationList, 
                 basedir='/home/puhao/dev/MAH/DataPreprocess/ODISE/demo/EPIC-KITCHEN') -> None:
        super(EpicKitchen, self).__init__()
        info = json.load(open(os.path.join(basedir, 'info.json'), 'r'))

        self.basedir = os.path.join(basedir, info[part][clip]['path'])
        self.out_basedir = os.path.join(basedir, part, 'mask_frames', clip)
        os.makedirs(self.out_basedir, exist_ok=True)
        self.total_count = info[part][clip]['count']

        self._init_transform(aug)

    def _init_transform(self, aug) -> None:
        self.aug = aug
        case = read_image(os.path.join(self.basedir, self._index_to_img_fn(0)), format="BGR")
        self.img_height, self.img_width = case.shape[:2]

    def _index_to_img_fn(self, index) -> str:
        return f'frame_{index + 1:0{self.LENGTH_FRAME_ID}d}.jpg'

    def __len__(self) -> int:
        return self.total_count

    def __getitem__(self, index) -> Tuple:
        img_path = os.path.join(self.basedir, self._index_to_img_fn(index))
        img = utils.read_image(img_path, format="RGB")
        aug_input = T.AugInput(img, sem_seg=None)
        self.aug(aug_input)
        img = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1))

        return {"image": img, "height": self.img_height, "width": self.img_width, 
                'out_path': os.path.join(self.out_basedir, self._index_to_img_fn(index))}
    
    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    cfg = LazyConfig.load("configs/Panoptic/odise_label_coco_50e.py")
    dataset_cfg = cfg.dataloader.test
    aug = instantiate(dataset_cfg.mapper).augmentations
    dataloader = EpicKitchen(part='P01', clip='P01_01',
                             aug=aug).get_dataloader(batch_size=4,
                                                     collate_fn=collate_fn_general,
                                                     num_workers=4,
                                                     pin_memory=True,
                                                     shuffle=False,)
    
    for i_b, batch in enumerate(dataloader):
        print(batch)