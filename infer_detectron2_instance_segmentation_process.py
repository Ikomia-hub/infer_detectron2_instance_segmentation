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
        self.custom_train = False
        self.cfg_path = ""
        self.weights_path = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.conf_thres = float(param_map["conf_thres"])
        self.cuda = eval(param_map["cuda"])
        self.custom_train = eval(param_map["custom_train"])
        self.cfg_path = param_map["cfg_path"]
        self.weights_path = param_map["weights_path"]
        self.update = utils.strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["cuda"] = str(self.cuda)
        param_map["custom_train"] = str(self.custom_train)
        param_map["cfg_path"] = self.cfg_path
        param_map["weights_path"] = self.weights_path
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2InstanceSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.predictor = None
        self.cfg = None
        self.class_names = None
        # Add instance segmentation output
        self.addOutput(dataprocess.CInstanceSegIO())

        # Create parameters class
        if param is None:
            self.setParam(InferDetectron2InstanceSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        self.forwardInputImage(0, 0)

        # Get parameters :
        param = self.getParam()
        if self.predictor is None or param.update:
            if param.custom_train:
                self.cfg = get_cfg()
                # add entry to cfg to avoid exception when merging from file
                self.cfg.CLASS_NAMES = None
                self.cfg.merge_from_file(param.cfg_path)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = param.weights_path
                self.class_names = self.cfg.CLASS_NAMES
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)
            else:
                self.cfg = get_cfg()
                config_path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs",
                                           param.model_name + '.yaml')
                self.cfg.merge_from_file(config_path)
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = param.conf_thres
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url((param.model_name + '.yaml').replace('\\', '/'))
                self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
                self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
                self.predictor = DefaultPredictor(self.cfg)

            param.update = False
            print("Inference will run on " + ('cuda' if param.cuda else 'cpu'))

        # Get input :
        img_input = self.getInput(0)
        instance_out = self.getOutput(1)

        if img_input.isDataAvailable():
            img = img_input.getImage()
            h, w, c = np.shape(img)
            instance_out.init("Detectron2_Instance_Segmentation", 0, w, h)
            colors = self.infer(img, instance_out)
            self.setOutputColorMap(0, 1, colors)
            self.forwardInputImage(0, 0)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

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
                    instance_out.addInstance(index, 0, cls, self.class_names[cls], float(score),
                                             float(x1), float(y1), w, h,
                                             mask.cpu().numpy().astype("uint8"), colors[cls+1])
                index += 1

        return colors

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2InstanceSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_instance_segmentation"
        self.info.shortDescription = "Infer Detectron2 instance segmentation models"
        self.info.description = "Infer Detectron2 instance segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.1.1"
        self.info.iconPath = "icons/detectron2.png"
        self.info.authors = "Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick"
        self.info.article = "Detectron2"
        self.info.journal = ""
        self.info.year = 2019
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentationLink = "https://detectron2.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/facebookresearch/detectron2"
        # Keywords used for search
        self.info.keywords = "infer, detectron2, instance, segmentation"

    def create(self, param=None):
        # Create process object
        return InferDetectron2InstanceSegmentation(self.info.name, param)
