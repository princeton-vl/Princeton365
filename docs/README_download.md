# Download from Huggingface


## Requirements
Please install the princeton365 package and (optionally) ffmpeg.

```bash
pip install princeton365
# (Optional) for unpacking stereo images
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

## ⚡ Quick Start

```bash
# Download and extract, but do not unpack stereo images (if you just need monocular sequences)
python scripts/download_princeton365.py

# Or Download, extract, and unpack everything in one command  
python scripts/download_princeton365.py --unpack-stereo

# Your data is ready in princeton365_data/
```

## 📁 Dataset Contents

Each sequence contains:
- `sequence_id.json`: Sequence metadata (experiment_id, category, split)
- `sequence_id.mp4`: User view video
- `sequence_id.user_camera_dist.npy`: User view camera distortion parameters
- `sequence_id.user_camera_mtx.npy`: User view camera intrinsic matrix
- `sequence_id.gt_camera_dist.npy`: GT view camera distortion parameters (validation only)
- `sequence_id.gt_camera_mtx.npy`: GT view camera intrinsic matrix (validation only)
- `sequence_id.relative_transform.npy`: Relative transformation from user to GT view
- `sequence_id.gt_trajectory.txt`: Ground truth camera trajectory (validation only)
- `sequence_id.gt_view.mp4`: Ground truth view video (validation only)
- `sequence_id.imu.csv`: IMU data
- `sequence_id.left_stereo.mp4`: Left stereo camera images (packed as MP4)
- `sequence_id.left_stereo_mapping.json`: Frame-to-timestamp mapping for left stereo images
- `sequence_id.right_stereo.mp4`: Right stereo camera images (packed as MP4)
- `sequence_id.right_stereo_mapping.json`: Frame-to-timestamp mapping for right stereo images
- `sequence_id.depth_frame_*.h5`: Downsampled depth data files for evaluation (validation only)
- `sequence_id.confidence_frame_*.h5`: Depth confidence files for evaluation (validation only)

## Download Options

### Extracted Format (Default - Recommended)

Extract files into organized directory structure for easy data loading:

```bash
# Download, extract, and unpack everything (default: extracts automatically)
python scripts/download_princeton365.py --unpack-stereo

# Custom output directory
python scripts/download_princeton365.py --unpack-stereo --output-dir ./my_data
```

### WebDataset Format 

Keep original webdataset tar files for use with webdataset loaders:

```bash
# Download as webdataset (tar files)
python scripts/download_princeton365.py --no-extract

# Custom output directory
python scripts/download_princeton365.py --no-extract --output-dir ./my_webdataset
```

### Single Sequence

```bash
# Download just one sequence (faster for testing)
python scripts/download_princeton365.py --sample 0 --unpack-stereo

# Download from different split
python scripts/download_princeton365.py --split test --sample 0
```


## Data Loading Examples


### Recommended: Using Extracted Format 

For simplest data loading, use the extracted format (default download):

```python
import os
import json
import numpy as np
import pandas as pd
from princeton365.utils.utils_depth import read_zed_depth

# After running: python download_princeton365.py --unpack-stereo
data_dir = "princeton365_data/validation"
sequences = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for sequence_id in sequences:
    seq_dir = os.path.join(data_dir, sequence_id)
    
    with open(f"{seq_dir}/{sequence_id}.json", 'r') as f:
        metadata = json.load(f)
        
    stereo_effective_fps = metadata.get('stereo_effective_fps')
    if stereo_effective_fps:
        print(f"Stereo effective FPS: {stereo_effective_fps}")
    
    user_camera_mtx = np.load(f"{seq_dir}/{sequence_id}.user_camera_mtx.npy")
    user_camera_dist = np.load(f"{seq_dir}/{sequence_id}.user_camera_dist.npy")
    
    gt_camera_path = f"{seq_dir}/{sequence_id}.gt_camera_mtx.npy"
    if os.path.exists(gt_camera_path):
        gt_camera_mtx = np.load(gt_camera_path)
        print(f"GT camera matrix shape: {gt_camera_mtx.shape}")
    
    rel_transform_path = f"{seq_dir}/{sequence_id}.relative_transform.npy"
    if os.path.exists(rel_transform_path):
        rel_transform = np.load(rel_transform_path)
        print(f"Relative transform shape: {rel_transform.shape}")
    
    imu_data = pd.read_csv(f"{seq_dir}/{sequence_id}.imu.csv")
    gt_trajectory = np.loadtxt(f"{seq_dir}/{sequence_id}.gt_trajectory.txt")
    
    depth_files = [f for f in os.listdir(seq_dir) if f.startswith(f"{sequence_id}.depth_frame_") and f.endswith('.h5')]
    if depth_files:
        depth_data = read_zed_depth(f)
        print(f"Depth data shape: {depth_data.shape}")

```


### Use WebDataset Library Directly

You can load from the tar files directly by using the `webdataset` library. First install webdataset with `pip install webdataset`. Then, you can load with

```python
import webdataset as wds

dataset = wds.WebDataset("princeton365_data/validation/000000.tar")

for sample in dataset:
    print("Available files:", list(sample.keys()))
    
    sequence_name = sample['__key__']           # e.g., 'new_scanning_72'
    json_metadata = sample['json']              # JSON metadata as bytes
    user_video = sample['mp4']                  # User view MP4 video as bytes
    gt_video = sample.get('gt_view.mp4')        # GT view MP4 video as bytes (if available)
    left_stereo = sample['left_stereo.mp4']     # Left stereo MP4 as bytes
    right_stereo = sample['right_stereo.mp4']   # Right stereo MP4 as bytes
    imu_data = sample['imu.csv']                # CSV data as bytes
    # ... other files
    
```



## File Structure

### Extracted Format (Default)

The data is organized by split and sequence for easy navigation:

```
princeton365_data/
└── validation/                                          # Split directory
    └── new_scanning_72/                                 # Sequence directory
        ├── new_scanning_72.json                         # Sequence metadata
        ├── new_scanning_72.mp4                          # User view video 
        ├── new_scanning_72.gt_view.mp4                  # GT view video 
        ├── new_scanning_72.user_camera_dist.npy         # User camera distortion parameters
        ├── new_scanning_72.user_camera_mtx.npy          # User camera intrinsic matrix
        ├── new_scanning_72.gt_camera_dist.npy           # GT camera distortion parameters 
        ├── new_scanning_72.gt_camera_mtx.npy            # GT camera intrinsic matrix 
        ├── new_scanning_72.relative_transform.npy       # Relative transformation user->GT
        ├── new_scanning_72.imu.csv                      # IMU data 
        ├── new_scanning_72.left_stereo.mp4              # Packed left stereo images 
        ├── new_scanning_72.left_stereo_mapping.json     # Left frame-to-filename mapping
        ├── new_scanning_72.right_stereo.mp4             # Packed right stereo images 
        ├── new_scanning_72.right_stereo_mapping.json    # Right frame-to-filename mapping
        ├── new_scanning_72.gt_trajectory.txt            # Ground truth trajectory 
        ├── new_scanning_72.depth_frame_0.h5             # Depth data files 
        ├── new_scanning_72.confidence_frame_0.h5        # Depth confidence files
        ├── left_stereo_images/                          # Unpacked left stereo images (if --unpack-stereo used)
        │   ├── 1739949778519433000.jpg                  # Original timestamp-named images
        │   ├── 1739949778536180000.jpg
        │   └── ... (3200+ total images)
        └── right_stereo_images/                         # Unpacked right stereo images (if --unpack-stereo used)
            ├── 1739949778519433000.jpg                  # Original timestamp-named images
            ├── 1739949778536180000.jpg
            └── ... (3200+ total images)
```

### WebDataset Format (--no-extract)

The data is kept in original webdataset format for ML pipelines:

```
princeton365_data/
└── validation/                          # Split directory
    ├── 000000.tar                       # Shard 0 (contains new_scanning_72 sequence)
    ├── 000001.tar                       # Shard 1 (contains next sequence)
    └── ...                              # Additional shards
```



