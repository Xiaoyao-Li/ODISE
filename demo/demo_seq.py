import argparse
import glob
import itertools
import numpy as np
import os
import tempfile
import time
import warnings
from contextlib import ExitStack
import cv2
import nltk
import json
import shutil
import torch
import tqdm

from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from torch import nn

from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.engine.defaults import get_model_from_module

from datasets.epic_kitchen import EpicKitchen
from datasets.misc import collate_fn_general

import yaml
from loguru import logger
from icecream import install
install()

nltk_path_local = '/home/puhao/.cache/nltk_data'

nltk.data.path.append(nltk_path_local)
nltk.download("popular", quiet=True, download_dir=nltk_path_local)
nltk.download("universal_tagset", quiet=True, download_dir=nltk_path_local)

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)

def frame_index_preprocess():
    assert False, 'Please confirm in script to run this function'
    PART = 'P01'
    CLIP = 'P01_01'
    START_FRAME = 1
    DATASET_INFO = json.load(open('./demo/EPIC-KITCHEN/info.json', 'r'))
    DATASET_BASEDIR = './demo/EPIC-KITCHEN/'
    CLIP_PATH = DATASET_INFO[PART][CLIP]['path']
    CLIP_COUNT = DATASET_INFO[PART][CLIP]['count']

    for frame_file in os.listdir(os.path.join(DATASET_BASEDIR, CLIP_PATH)):
        ori_path = os.path.join(DATASET_BASEDIR, CLIP_PATH, frame_file)
        tar_path = os.path.join(DATASET_BASEDIR, CLIP_PATH, f'frame_{int(frame_file[6:-4]) - START_FRAME + 1:010d}.jpg')
        shutil.move(ori_path, tar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for EPIC-KITCHEN")
    parser.add_argument(
        "--config-file",
        default="configs/Panoptic/odise_label_coco_50e.py",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        help="init from the given checkpoint",
        default="odise://Panoptic/odise_label_coco_50e",
    )
    parser.add_argument(
        "--vocab",
        help="extra vocabulary, in format 'a1,a2;b1,b2',"
        "where a1,a2 are synonyms vocabularies for the first class"
        "first word will be displayed as the class name",
        default="person, child, girl, boy, woman, man, perple, children,\
              girls, boys, women, men, lady, ladies, guys",
    )
    parser.add_argument(
        "--label",
        help="label set to use, could be multiple options from 'COCO', 'ADE' and 'LVIS'.",
        choices=["COCO", "ADE", "LVIS", ""],
        nargs="+",
        default="",
    )
    args = parser.parse_args()

    logger.info("Arguments Configuration: " + '\n' + yaml.dump(vars(args)))

    cfg = LazyConfig.load(args.config_file)
    cfg.model.overlap_threshold = 0
    cfg.model.clip_head.alpha = 0.35
    cfg.model.clip_head.beta = 0.65
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    extra_classes = []

    if args.vocab:
        for words in args.vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])

    logger.info(f"extra classes: {extra_classes}")
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []
    
    if "COCO" in args.label:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors = COCO_STUFF_COLORS
    if "ADE" in args.label:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in args.label:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS
    
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    wrapper_cfg.labels = demo_thing_classes + demo_stuff_classes
    wrapper_cfg.metadata = demo_metadata

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(args.init_from)

    while "model" in wrapper_cfg:
        wrapper_cfg = wrapper_cfg.model
    wrapper_cfg.model = get_model_from_module(model)

    dataloader = EpicKitchen(part='P01', clip='P01_01',
                             aug=aug).get_dataloader(batch_size=4,
                                                     collate_fn=collate_fn_general,
                                                     num_workers=4,
                                                     pin_memory=True,
                                                     shuffle=False,
                                                    )
    
    inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
    with ExitStack() as stack:
        if isinstance(inference_model, nn.Module):
            stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        for i_b, batch in enumerate(dataloader):
            ic(batch[0]['out_path'])
            img = batch[0]['image']
            img = img.to('cpu').numpy()
            # np.save('demo/img_demo_seq.npy', img)
            ic(img)

            st = time.time()
            predictions = inference_model(batch)
            ic(time.time() - st)

            pred = predictions[0]
            pred = pred['panoptic_seg'][0].to('cpu').numpy()
            ic(pred)
            # np.save('demo/demo_seq.npy', np.concatenate([img.astype('int32'), np.expand_dims(pred, axis=0)], axis=0))

            for i_pred, pred in enumerate(predictions):
                panoptic_seg, segments_info = pred['panoptic_seg']
                panoptic_seg = panoptic_seg.to('cpu').numpy()
                for panoptic_label in np.unique(panoptic_seg):
                    label_seg = (panoptic_seg == panoptic_label)

                    import matplotlib.pyplot as plt
                    plt.imshow(label_seg, cmap='binary', interpolation=None)
                    plt.title(f'i_pred: {i_pred}')
                    plt.show()
                    # ic()

                ic(batch[i_pred])
                
            ic(len(predictions))
