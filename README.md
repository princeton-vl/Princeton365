<p align="center">

  <h1 align="center">Princeton365: A Diverse Dataset with Accurate Camera Pose</h1>
  <p align="center">
    <a href="https://kkayan.com/"><strong>Karhan Kayan*</strong></a>
    ·
    <a href="https://stamatisalex.github.io/"><strong>Stamatis Alexandropoulos*</strong></a>
    ·
    <a href="https://www.linkedin.com/in/rishabhj0"><strong>Rishabh Jain</strong></a>
    ·
    <a href="https://zuoym15.github.io/"><strong>Yiming Zuo</strong></a>
    ·
    <a href="https://www.linkedin.com/in/erlian"><strong>Erich Liang</strong></a>    
    .
    <a href="https://www.cs.princeton.edu/~jiadeng/"><strong>Jia Deng</strong></a>    
  </p>
  <p align="center">
    (*equal contribution, random order)
  </p>
  <h4 align="center">
  Princeton University    
  </h4>
</p>

<h3 align="center"><a href="https://princeton365.cs.princeton.edu/">Website</a> | <a href="https://arxiv.org/abs/2506.09035">Paper</a> </a></h3>

<p align="center">
  <a href="https://arxiv.org/abs/2410.10799">
    <img src="./media/main_fig.png" alt="Logo" width="98%">
  </a>
</p>


## Citation
If you use our benchmark, data, or method in your work, please cite our paper:
```
@misc{princeton365,
      title={Princeton365: A Diverse Dataset with Accurate Camera Pose}, 
      author={Karhan Kayan and Stamatis Alexandropoulos and Rishabh Jain and Yiming Zuo and Erich Liang and Jia Deng},
      year={2025},
      eprint={2506.09035},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.09035}, 
}
```

## Downloading the Dataset
Please download the dataset from our [website](https://princeton365.cs.princeton.edu/download/). The test set does not contain ground-truth and should be submitted to our evaluation server.  



## Evaluating Your SLAM/Odometry Method

### Installation

For basic functionality (submitting results):
```bash 
conda create --name princeton365 python=3.10
conda activate princeton365
pip install .
```

For development with ground truth generation capabilities:
```bash
conda create --name princeton365 python=3.10
conda activate princeton365
pip install ".[dev]"
```

You can also install the package from PyPI:
```bash
pip install princeton365
```
or 
```bash 
pip install princeton365[dev]
```


### Run Your Method

To evaluate your model on the test set, you need to submit your results to our evaluations server. Please run your model on the test set and convert the trajectories into **TUM Format**. This means that the resulting trajectories for each sequence should be a .txt file with `timestamp tx ty tz qx qy qz qw` on each line. See [here](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) for details. Put all of your estimated trajectories in the same folder. The trajectories should have exactly the same name as the corresponding video (i.e.  For the video new_scanning_user_view_1.mp4 the trajectory should be named new_scanning_user_view_1.txt)


### Submit Your Results

Submit your predictions to the evaluation server using the command below. Replace the placeholders:

```bash
princeton365-upload \
    --email your_email \
    --path path_to_your_submission_folder \
    --method_name your_method_name
```

### After Submission

Upon submission, you will receive a unique submission ID, which serves as the identifier for your submission. Results are typically emailed within a few hours. Please note that each email user may upload only three submissions every seven days.

### Making Your Submission Public

To make your submission public, run:

```bash
princeton365-make-public \
    --id submission_id \
    --email your_email \
    --anonymous False \
    --method_name your_method_name \
    --publication "your publication name" \
    --url_publication "https://your_publication" \
    --url_code "https://your_code" \
```

You may set `"Anonymous"` as the publication name if the work is under review. The `url_publication`, `url_code` fields are optional. If your method uses IMU data, please add `--inertial` flag. If your method uses stereo data, please add `--stereo` flag. 

## Evaluating Your NVS Method
The NVS benchmark is available on our [website](https://princeton365.cs.princeton.edu/download) with full ground-truth, and it does not require submission. We recommend installing [Nerfstudio](https://docs.nerf.studio/) to run your method, and we provide our ground-truth in COLMAP format as well, which is required by Nerfstudio. 

Examples of training and evaluation scripts are provided in the `scripts/nvs` folder. For instance, to train and evaluate a NeRF model, you can run:
```bash
bash scripts/nvs/train_nerf_all.sh
bash scripts/nvs/test_all.sh
```

Note that the sequences in the cross-validation set of the SLAM benchmark can also be used to evaluate NVS methods since they contain the camera pose ground-truth. 


# Using the Ground-truth Method
Princeton365 uses a method for camera pose ground-truth that allows for easy data collection with high accuracy. In this section, we give instructions on how you can annotate your own sequences with our ground-truth method. Please see [the website](https://princeton365.cs.princeton.edu/setup/) for our camera rig setup. You do not need an exact replica of this setup to start collecting data. For monocular camera pose, you only need a 360-camera or two cameras with fixed relative pose. 

If you want to use the Induced Optical Flow metric, you will need depth information, which you can get using a stereo camera or a Lidar device. **This repo is only for producing ground-truth trajectories and the optical-flow metric and not for generating the depth data.**


### Installation


Follow the steps below to set up the required environment:

```bash
conda create --name princeton365
conda activate princeton365
pip install ".[dev]"
sudo apt-get install python3-tk
sudo apt-get install python3-pil.imagetk
```

You need a **different environment** to run Bundle PnP. Please install the following: 
```bash 
conda create --name bundle_pnp 
conda activate bundle_pnp
conda install -c conda-forge ceres-solver cxx-compiler c-compiler
conda install -c conda-forge zlib
cd princeton365/optimization/bundle_pnp
mkdir build
cd build
cmake -DMETIS_LIBRARY=/path/to/libmetis.so -DMETIS_INCLUDE_DIR=/path/to/metis/include ..
make
cd ../../../..
```

To install Bundle Rig PnP to compute relative pose between two views, run the following commands: 
```bash 
cd princeton365/optimization/rig_bundle_pnp
mkdir build
cd build
cmake -DMETIS_LIBRARY=/path/to/libmetis.so -DMETIS_INCLUDE_DIR=/path/to/metis/include ..
make
cd ../../../..
```


#### Handling GCC Compiler Errors

If you encounter issues with the GCC compiler, add the following aliases to your `~/.bashrc` file:

```bash
alias gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
alias g++="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
```

Then, reload the `.bashrc` file:

```bash
source ~/.bashrc
```


### File Structure

- **Benchmark Videos** (`pvl-slam/Benchmark`):
    - Contains Benchmark videos (scanning, indoor, outdoors)
    - Each folder includes the following three subdirectories:
        - `gt_view`:Ground truth (GT) view videos, the corresponding user_view video of which will be used in evaluations.
        - `pose_graph_extra_frames`: GT view videos used for close-up filming and pose graph construction
        - `user_view`: User-perspective videos that correspond to the GT view videos.


### Generating Calibration Boards

To generate the calibration boards (either grid or ChArUco), use the following command:

```bash
conda activate princeton365
princeton365-board-generator --config princeton365/configs/board_configs.yaml --board_type '<type>'
```
Replace <type> with the desired board type, such as 'grid' or 'charuco'. You can customize the board parameters by editing the `princeton365/configs/board_configs.yaml` file to suit your needs.

Print these boards and place them completely flat on a surface. The local 3D board coordinates assume that the markers are on a flat plane. This can be ensured cheaply by gluing the paper boards on a flat cardboard. Higher precision techniques such as printing on ceramic or aluminum surfaces can also be used. 

## Generating Ground Truth (GT) Trajectories

The ground truth trajectory is generated in two main steps:

1. **Generate the Pose Graph from a Close-Up Video**:

Run the following command on a close-up video of the calibration board to generate a pose graph and initial trajectory:

```bash
conda activate princeton365
princeton365-generate-gt \
    --video <close_up_video> \
    --intrinsics <intrinsics_folder> \
    --board_type <'grid' or 'charuco'>
```
where: 
 - `<close_up_video>`: Path to the close-up video capturing the calibration board
 - `<intrinsics_folder>`: Directory containing camera intrinsics
 - `<'grid' or 'charuco'>`: Type of calibration board used


2. **Generate the GT Trajectory for the Main View:**
Use the pose graph from the close-up video to generate the trajectory for the main GT view:

```bash
conda activate princeton365
princeton365-generate-gt \
    --video <gt_view_video> \
    --intrinsics <intrinsics_folder> \
    --board_type <'grid' or 'charuco'> \
    --use_pose_graph <pose_graph_file> \
    --bundle_pnp
```
where
- `<gt_view_video>`: Path to the GT view video
- `<pose_graph_file>`: Pose graph file generated from the close-up video
- `<intrinsics_folder>` and --board_type should match the setup used in Step 1



3. **Relative Pose Generator**

This utility script finds the relative pose between two views (e.g., ground-truth and user views) using detected 2D points, camera intrinsics, and a ground-truth trajectory. It uses Bundle Rig PnP. 

```bash
conda activate princeton365
princeton365-relative-pose \
    --gt_trajectory <path_to_gt_trajectory_txt> \
    --gt_detected_points <path_to_gt_detected_points_json> \
    --user_detected_points <path_to_user_detected_points_json> \
    --gt_intrinsics <path_to_gt_intrinsics_folder> \
    --user_intrinsics <path_to_user_intrinsics_folder> \
    --pose_graph <path_to_pose_graph_pickle>
```


