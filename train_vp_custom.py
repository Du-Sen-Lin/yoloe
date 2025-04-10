from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_seg_vp import YOLOESegVPTrainer
from ultralytics.models.yolo.yoloe.train_vp import YOLOEVPTrainer
from ultralytics.models.yolo.yoloe import YOLOETrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER

os.environ["PYTHONHASHSEED"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# data = dict(
#     train=dict(
#         yolo_data=["/root/project/research/Yolo/yoloe/ultralytics/cfg/datasets/custom.yaml"],
#     ),
#     val=dict(yolo_data=["/root/project/research/Yolo/yoloe/ultralytics/cfg/datasets/custom.yaml"]),
# )

data = dict(
    train=dict(
        yolo_data=["/root/project/research/Yolo/yoloe/ultralytics/cfg/datasets/glass.yaml"],
    ),
    val=dict(yolo_data=["/root/project/research/Yolo/yoloe/ultralytics/cfg/datasets/glass_train_vps.yaml"]),
)

model_path = "yoloe-v8l.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOE("./pretrain/yoloe-v8l-seg-det.pt")
# YOLOESegVPTrainer => YOLOEVPTrainer

head_index = len(model.model.model) - 1
freeze = list(range(0, head_index))
for name, child in model.model.model[-1].named_children():
    if 'savpe' not in name:
        freeze.append(f"{head_index}.{name}")

# For s/m, please set lr0=8e-3
model.train(data=data, batch=32, epochs=100, **extends, close_mosaic=2, \
    optimizer='AdamW', lr0=16e-3, warmup_bias_lr=0.0, \
        weight_decay=0.025, momentum=0.9, workers=4, \
        trainer=YOLOEVPTrainer, device='0', freeze=freeze, load_vp=True)

# FileNotFoundError: [Errno 2] No such file or directory: 'tools/mobileclip:blt/global_grounding_neg_embeddings.pt'
# model.train(data=data, batch=32, epochs=2, **extends, close_mosaic=2, \
#     optimizer='AdamW', lr0=16e-3, warmup_bias_lr=0.0, \
#         weight_decay=0.025, momentum=0.9, workers=4, \
#         trainer=YOLOEVPTrainer, device='0', freeze=freeze, load_vp=True)

# model.train(data="/root/project/research/Yolo/yoloe/ultralytics/cfg/datasets/custom.yaml", batch=32, epochs=2, **extends, close_mosaic=2, \
#     optimizer='AdamW', lr0=16e-3, warmup_bias_lr=0.0, \
#         weight_decay=0.025, momentum=0.9, workers=4, \
#         trainer=YOLOETrainer, device='0', freeze=freeze, load_vp=False)