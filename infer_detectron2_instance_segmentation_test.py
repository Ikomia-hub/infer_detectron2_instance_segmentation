from ikomia.core import task, ParamMap
import ikomia
import os
import yaml
import cv2
import detectron2


def test(t, data_dict):
    img = cv2.imread(data_dict["images"]["detection"]["text"])[::-1]
    input_img = t.getInput(0)
    input_img.setImage(img)

    config_paths = os.path.dirname(detectron2.__file__) + "/model_zoo"

    for root, dirs, files in os.walk(config_paths, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            possible_cfg = os.path.join(*file_path.split('/')[-2:])
            if "InstanceSegmentation" in possible_cfg or "Cityscapes" in possible_cfg and possible_cfg.endswith(
                    '.yaml'):
                params = task.get_parameters(t)
                params["model_name"] = possible_cfg.replace('.yaml', '')
                # without update = 1, model is not updated between 2 test
                params["update"] = 1
                task.set_parameters(t, params)
                t.run()


