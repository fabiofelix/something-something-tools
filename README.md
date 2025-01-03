

This repo has codes that can extend [Something-something](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) dataset.
Check some outputs in the *examples* folder.

# **Automatic object segmentation**

[![Cuda 11.8.86](https://img.shields.io/badge/cuda-11.8.86-red.svg)]() [![cuDNN 8.7](https://img.shields.io/badge/cudnn-8.7-blue.svg)]() [![Python 3.12.8](https://img.shields.io/badge/python-3.12.8-green.svg)](https://www.python.org/downloads/release/python-3128/) 


It uses [Segment-Anything](https://github.com/facebookresearch/segment-anything) model to segment the objects inside bounding boxes annotated by
[Something-else](https://github.com/joaanna/something_else) project.

Before executing the code, one should download the [Something-something](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads) data, the [Something-else](https://github.com/joaanna/something_else) annotations, and install the following requirements. 

  ```
  pip install -r requirements_seg.txt
  ```

The code uses the [ViT-H SAM](https://github.com/facebookresearch/segment-anything) model checkpoint to perform the segmentation. It should be downloaded from the link and saved in the same code folder.
It is possible to use another checkpoint by downloading it and changing the **Segment_MyVideo** *model_type* and *sam_checkpoint* parameters.

Following, there are running examples to generate the masks and visualize the segment results.

  ```
  python segment_dataset.py -a seg -s /path/to/something/something/video/frames/ -o /path/to/output/masks/ -f 0
  ```

  ```
  python segment_dataset.py -a load -s /path/to/something/something/video/frames  -m /path/to/saved/masks/ -o /path/to/output/video/segment/visualization/
  ```

> [!NOTE] 
> The bounding box annotations are split into four files and the *-f* argument provides an interface to process each of them separately with values varying in the [0, 3] range.


# **Associate data with first-order logic descriptions**

[![Cuda 11.8.86](https://img.shields.io/badge/cuda-11.8.86-red.svg)]() [![cuDNN 8.7](https://img.shields.io/badge/cudnn-8.7-blue.svg)]() [![Python 3.12.8](https://img.shields.io/badge/python-3.12.8-green.svg)](https://www.python.org/downloads/release/python-3128/)


A dataloader that combines [Something-something](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads) data, the [Something-else](https://github.com/joaanna/something_else) annotations, and [PDDL](https://gist.githubusercontent.com/beasteers/defa94fb90a66a14b279b9b69b23f0fc/raw/5cd28cfe45e3e251e7dac40b0e13959160a01b43/domain_20bn.pddl) definitions of video actions.
This dataloader can be used to train models that reason about objections, actions, and environments such as [link](https://ieeexplore.ieee.org/abstract/document/9812016).

Before executing the code, one should download the [Something-something](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads) data, the [Something-else](https://github.com/joaanna/something_else) annotations, the [PDDL](https://gist.githubusercontent.com/beasteers/defa94fb90a66a14b279b9b69b23f0fc/raw/5cd28cfe45e3e251e7dac40b0e13959160a01b43/domain_20bn.pddl) definitions, and install the following requirements. 

  ```
  pip install -r requirements_loader.txt
  ```

A simple use of the defined dataloader

  ```
  from torch.utils.data import DataLoader
  from dataloader import CustomVideoDataset

  data   = CustomVideoDataset(main_video_dir = "/path/to/something_else/video/frames", 
                              annotations_file = "/path/to/something_else/annotation.json", 
                              labels_dir = "/path/to/something_something/bounding/box/annotations", 
                              pddl_file = "/path/to/pddl/domain_20bn.pddl",
                              return_frame = "first_last",
                              resize_shape=(240, 320))
  loader = DataLoader(data, batch_size=2, shuffle=False) 

  for video_id, frames, boxes, surroundes, categories, precondition_obj, precondition_rel, effect_obj, effect_rel in loader:  
    print(f"========================== TESTING {video_id}==========================")
    ...
  ```

For each frame (*frame_name*) in a video (*video_id*) that has bounding boxes annotated, it returns:

1. from *Something-something*: min-max normalized RGB frame images. It could also return a reshaped version (resize_shape is not None)
1.1 It can return all the video frames (*return_frame* = RETURN_ALL_FRAMES) or only the first-and-last frames (*return_frame* = RETURN_ONLY_FIRST_AND_LAST), the default behaviour  
2. from Something-else based on *video_id* and *frame_name*: bounding box (x1, y1, x2, y2) and category (integer-encoded, description)
3. from PDDL definitions based on *video_id*, *Something-something placeholders*, and *Something-else bounding box category description*: lists of action pre-conditions and effects (post-conditions)
3.1 The first half of the list has all possible predicates. Each position describes whether the predicate is AFFIRMATIVE, NEGATIVE, OR NONAPPLICABLE
3.2 The second half of the list has the inverse for AFFIRMATIVE and NEGATIVE predicates.
3.3 All possible predicates are listed in the properties *pddl_domain_predicates_single* and *pddl_domain_predicates_double*.
4. It also returns a bounding box with all objects present in Something-else and PDDL definitions.
