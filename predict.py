# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import io
import base64
import cv2
import tensorflow as tf
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page
from PIL import Image
from cog import BasePredictor, Input

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

DET_ARCHS = ["db_resnet50", "db_mobilenet_v3_large"]
RECO_ARCHS = ["crnn_vgg16_bn", "crnn_mobilenet_v3_small", "master", "sar_resnet31"]

class Predictor(BasePredictor):
    def setup(self):
        self.det_arch = DET_ARCHS[0] # Choose default models
        self.reco_arch = RECO_ARCHS[0] # Choose default models

    def predict(self,
        image: str = Input(description="Input base64 image string to query or caption"),
        det_arch: str = Input(description="Text detection model", default=DET_ARCHS[0]),
        reco_arch: str = Input(description="Text recognition model", default=RECO_ARCHS[0]),
        ):
        """Run a single prediction on the model"""
        # convert the base64 string to bytes
        image_data = base64.b64decode(image)
        # create a bytes stream for PIL
        image_stream = io.BytesIO(image_data)
        # load the image from the stream
        raw_image = Image.open(image_stream).convert("RGB")

        doc = DocumentFile.from_images([raw_image])

        self.det_arch = det_arch
        self.reco_arch = reco_arch

        predictor = ocr_predictor(self.det_arch, self.reco_arch, pretrained=True)

        # Forward the image to the model
        processed_batches = predictor.det_predictor.pre_processor([doc[0]])
        out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
        seg_map = out["out_map"]
        seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
        seg_map = cv2.resize(seg_map.numpy(), (doc[0].shape[1], doc[0].shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Perform OCR
        out = predictor([doc[0]])

        # Page reconsitution under input page
        page_export = out.pages[0].export()

        return page_export