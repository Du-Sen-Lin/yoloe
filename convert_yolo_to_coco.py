import os
import json
from pathlib import Path
from PIL import Image

def yolo_to_coco(image_dir, label_dir, output_path, class_names):
    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, Path(img_file).with_suffix('.txt'))

        # 获取图像宽高
        with Image.open(img_path) as img:
            width, height = img.size

        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id, x, y, w, h = map(float, line.strip().split())
                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height
                    box_w = w * width
                    box_h = h * height
                    area = box_w * box_h

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls_id),
                        "bbox": [x1, y1, box_w, box_h],
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_path, 'w') as f:
        json.dump(coco_dict, f)
    print(f"Saved COCO annotations to {output_path}")

# 示例调用
if __name__ == "__main__":
    yolo_to_coco(
        image_dir="/root/dataset/glass_data_20250317/images/train",
        label_dir="/root/dataset/glass_data_20250317/labels/train",
        output_path="custom_train.json",
        class_names=["Bracket component", "Sliding sleeve", "Elastic bridge", "Fastening screw", "Fastening nut", "Small adjustment block", "Large adjustment block"]  # 根据你的数据集修改
    )
