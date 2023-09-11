<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_detectron2_instance_segmentation/main/icons/detectron2.png" alt="Algorithm icon">
  <h1 align="center">infer_detectron2_instance_segmentation</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_detectron2_instance_segmentation">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_detectron2_instance_segmentation">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_detectron2_instance_segmentation/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_detectron2_instance_segmentation.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run Detectron2 instance segmentation models. It can detect and segment objects in image.

![Example](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*G5EsdDTv9-5kqK0hu9fIJw.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

[Change the sample image URL to fit algorithm purpose]

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_detectron2_instance_segmentation", auto_connect=True)

# Run on your image
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Display the results
display(algo.get_image_with_mask_and_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - Default "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x": Name of the pretrained model. 
Should be one of:
  - COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x
  - COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x
  - COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x
  - COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x
  - LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x
  - LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x
  - LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x
  - Cityscapes/mask_rcnn_R_50_FPN
- **conf_thres** (float) - Default 0.5: Box threshold for the prediction [0,1].
- **cuda** (bool) - Default True: If True, CUDA-based inference (GPU). If False, run on CPU.
- **config_file** (str): Path to the .yaml config file. Overwrite model_name if both are provided.
- **model_weight_file** (str): Path to the .pth weight file. Overwrite model_name if both are provided.

*Note*: parameter key and value should be in **string format** when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_detectron2_instance_segmentation", auto_connect=True)

# Set parameters
algo.set_parameters({
    "model_name": "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x",
})

# Run on your image
wf.run_on(url="https://cdn.nba.com/teams/legacy/www.nba.com/bulls/sites/bulls/files/jordan_vs_indiana.jpg")

# Display the results
display(algo.get_image_with_mask_and_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_detectron2_instance_segmentation", auto_connect=True)

# Run on your image  
wf.run_on(url="example_image.png")

# Iterate over outputs
for output in algo.get_outputs()
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```