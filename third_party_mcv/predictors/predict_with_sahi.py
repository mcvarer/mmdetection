import os

from sahi.model import MmdetDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.mmdet import (
    download_mmdet_cascade_mask_rcnn_model,
    download_mmdet_config,
)

model_path = 'models/cascade_mask_rcnn.pth'
download_mmdet_cascade_mask_rcnn_model(model_path)
config_path = download_mmdet_config(model_name="cascade_rcnn", config_file_name="cascade_mask_rcnn_r50_fpn_1x_coco.py",)

detection_model = MmdetDetectionModel(
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.4,
    device="cuda:0",  # or 'cuda:0'
)

__inputDIR__ = os.path.join('third_party_mcv', 'predictors', 'test_images')
__outputDIR__ = os.path.join('third_party_mcv', 'predictors', 'exports')
if not __outputDIR__:
    os.makedirs(__outputDIR__)

img_name = 'CPLX_G0010768.jpg'
img = os.path.join(__inputDIR__, f'{img_name}')  # or img = mmcv.imread(img), which will only load it once

result = get_prediction(img, detection_model)

result.export_visuals(export_dir=os.path.join(__outputDIR__), file_name="not_slice")

result = get_sliced_prediction(
    image=img,
    detection_model=detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
result.export_visuals(export_dir=os.path.join(__outputDIR__), file_name="with_slice")
