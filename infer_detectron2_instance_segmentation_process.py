# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from infer_detectron2_instance_segmentation import update_path
from ikomia import utils, core, dataprocess
import copy
import os
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import numpy as np
import torch


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDetectron2InstanceSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x"
        self.conf_thres = 0.5
        self.cuda = True if torch.cuda.is_available() else False
        self.update = False
        self.use_custom_model = False
        self.config_file = ""
        self.model_weight_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.cuda = eval(param_map["cuda"])
        self.use_custom_model = eval(param_map["use_custom_model"])
        self.config_file = param_map["config_file"]
        self.model_weight_file = param_map["model_weight_file"]
        self.update = utils.strtobool(param_map["update"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": self.model_name,
            "conf_thres": str(self.conf_thres),
            "cuda": str(self.cuda),
            "use_custom_model": str(self.use_custom_model),
            "config_file": self.config_file,
            "model_weight_file": self.model_weight_file,
            "update": str(self.update)
            }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2InstanceSegmentation(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        # Add input/output of the process here
        self.predictor = None
        self.cfg = None
        self.class_names = None
        # Add instance segmentation output
        #self.add_output(dataprocess.CInstanceSegIO())

        # Create parameters class
        if param is None:
            self.set_param_object(InferDetectron2InstanceSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()
        self.forward_input_image(0, 0)

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["FVCORE_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

        if self.predictor is None or param.update:
            if param.model_weight_file != "":
                if os.path.isfile(param.model_weight_file):
                    param.use_custom_model = True

            if param.use_custom_model:
                self.cfg = get_cfg()
                # add entry to cfg to avoid exception when merging from file
                self.cfg.CLASS_NAMES = None
                self.cfg.merge_from_file(param.config_file)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = param.model_weight_file
                self.class_names = self.cfg.CLASS_NAMES
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda and torch.cuda.is_available() else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)
            else:
                self.cfg = get_cfg()
                config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                           param.model_name + '.yaml')
                self.cfg.merge_from_file(config_path)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name + '.yaml').replace('\\', '/'))
                self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda and torch.cuda.is_available() else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)

            param.update = False
            print("Inference will run on " + self.cfg.MODEL.DEVICE)

        self.set_names(self.class_names)
        # Get input :
        img_input = self.get_input(0)
        instance_out = self.get_output(1)

        if img_input.is_data_available():
            img = img_input.get_image()
            colors = self.infer(img, instance_out)
            self.forward_input_image(0, 0)

        os.environ.pop("FVCORE_CACHE")

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def infer(self, img, instance_out):
        outputs = self.predictor(img)

        if "instances" in outputs.keys():
            instances = outputs["instances"].to("cpu")
            scores = instances.scores
            boxes = instances.pred_boxes
            classes = instances.pred_classes
            masks = instances.pred_masks

            np.random.seed(10)
            colors = [[0, 0, 0]]
            for i in range(len(self.class_names)):
                colors.append([int(c) for c in np.random.choice(range(256), size=3)])

            index = 0
            for box, score, cls, mask in zip(boxes, scores, classes, masks):
                if score >= self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                    x1, y1, x2, y2 = box.numpy()
                    cls = int(cls.numpy())
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    self.add_object(index, 0, cls, float(score), float(x1), float(y1), w, h,
                                    mask.cpu().numpy().astype("uint8"))
                index += 1

        return colors


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2InstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_instance_segmentation"
        self.info.short_description = "Infer Detectron2 instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.3.1"
        self.info.icon_path = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_detectron2_instance_segmentation"
        self.info.original_repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, instance, segmentation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INSTANCE_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferDetectron2InstanceSegmentation(self.info.name, param)
