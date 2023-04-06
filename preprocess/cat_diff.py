from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from odise.data import get_openseg_labels
from icecream import install
install()

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]

COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]

ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)


if __name__ == '__main__':
    ic(COCO_THING_CLASSES)
    ic(COCO_STUFF_CLASSES)
    ic(ADE_THING_CLASSES)
    ic(ADE_STUFF_CLASSES)
    ic(LVIS_CLASSES)
    

