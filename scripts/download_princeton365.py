#!/usr/bin/env python3

"""
Princeton365 Dataset Downloader

A comprehensive tool to download, extract, and unpack Princeton365 datasets from HuggingFace.
"""

import argparse
import os
import sys
import tarfile
import subprocess
import shutil
from huggingface_hub import hf_hub_download, list_repo_files

def list_available_shards(repo_id, split="validation"):
    """List all available shards in the repository"""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        shard_files = [f for f in files if f.startswith(f"{split}/") and f.endswith(".tar")]
        shard_indices = []
        for f in shard_files:
            filename = os.path.basename(f)
            if filename.replace(".tar", "").isdigit():
                shard_indices.append(int(filename.replace(".tar", "")))
        return sorted(shard_indices)
    except Exception as e:
        print(f"❌ Error listing repository files: {e}")
        return []

def discover_available_splits(repo_id):
    """Discover all available splits in the repository"""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        splits = set()
        for f in files:
            if f.endswith(".tar") and "/" in f:
                split_name = f.split("/")[0]
                splits.add(split_name)
        return sorted(list(splits))
    except Exception as e:
        print(f"❌ Error discovering splits: {e}")
        return []

def download_shard(repo_id, shard_idx, split, cache_dir=None):
    """Download a single shard"""
    tar_filename = f"{shard_idx:06d}.tar"
    print(f"📥 Downloading {split}/{tar_filename}...")
    
    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"{split}/{tar_filename}",
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        file_size_mb = os.path.getsize(downloaded_file) / (1024*1024)
        print(f"✅ Downloaded: {file_size_mb:.1f} MB")
        return downloaded_file
        
    except Exception as e:
        print(f"❌ Error downloading {tar_filename}: {e}")
        return None

def extract_shard(tar_path, output_dir, split):
    """Extract a shard to organized directory structure (split/sequence_id/)"""
    print(f"📦 Extracting {os.path.basename(tar_path)}...")
    
    try:
        # First extract to temporary location to examine contents
        temp_dir = os.path.join(output_dir, "temp_extract")
        os.makedirs(temp_dir, exist_ok=True)
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # Find sequences by grouping files by prefix
        temp_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        sequences = {}
        for file in temp_files:
            if '.' in file:
                prefix = file.split('.')[0]
                if prefix not in sequences:
                    sequences[prefix] = []
                sequences[prefix].append(file)
        
        # Move files to organized structure: split/sequence_id/files
        total_files = 0
        for sequence_id, files in sequences.items():
            sequence_dir = os.path.join(output_dir, split, sequence_id)
            os.makedirs(sequence_dir, exist_ok=True)
            
            for file in files:
                src = os.path.join(temp_dir, file)
                dst = os.path.join(sequence_dir, file)
                shutil.move(src, dst)
                total_files += 1
            
            print(f"  📁 {sequence_id}: {len(files)} files → {sequence_dir}")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"✅ Extracted {total_files} files in organized structure")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {tar_path}: {e}")
        return False

def find_sequences(data_dir):
    """Find all sequences in the organized directory structure (split/sequence_id/)"""
    if not os.path.exists(data_dir):
        return {}
    
    sequences = {}
    # Look for split directories
    for split_dir in os.listdir(data_dir):
        split_path = os.path.join(data_dir, split_dir)
        if os.path.isdir(split_path):
            # Look for sequence directories within split
            for sequence_dir in os.listdir(split_path):
                sequence_path = os.path.join(split_path, sequence_dir)
                if os.path.isdir(sequence_path):
                    # Get files in this sequence directory
                    files = [f for f in os.listdir(sequence_path) 
                            if os.path.isfile(os.path.join(sequence_path, f))]
                    if files:
                        sequence_key = f"{split_dir}/{sequence_dir}"
                        sequences[sequence_key] = files
    
    return sequences

def unpack_stereo_video(data_dir, sequence_key):
    """Unpack stereo MP4 back to individual JPG images"""
    # sequence_key is like "validation/new_scanning_72"
    split_dir, sequence_id = sequence_key.split('/', 1)
    sequence_dir = os.path.join(data_dir, split_dir, sequence_id)
    
    # Check for both left and right stereo videos
    left_stereo_mp4 = os.path.join(sequence_dir, f"{sequence_id}.left_stereo.mp4")
    left_mapping_file = os.path.join(sequence_dir, f"{sequence_id}.left_stereo_mapping.json")
    
    right_stereo_mp4 = os.path.join(sequence_dir, f"{sequence_id}.right_stereo.mp4")
    right_mapping_file = os.path.join(sequence_dir, f"{sequence_id}.right_stereo_mapping.json")
    
    # Check if pack_stereo.py exists
    pack_stereo_script = "scripts/pack_stereo.py"
    if not os.path.exists(pack_stereo_script):
        print(f"❌ pack_stereo.py script not found at {pack_stereo_script}")
        print("   Please ensure the script is available in the scripts/ directory")
        return False
    
    print(f"🎬 Unpacking stereo videos for {sequence_key}...")
    success = False
    
    # Unpack left stereo video
    if os.path.exists(left_stereo_mp4) and os.path.exists(left_mapping_file):
        left_stereo_images_dir = os.path.join(sequence_dir, "left_stereo_images")
        os.makedirs(left_stereo_images_dir, exist_ok=True)
        
        try:
            cmd = [
                sys.executable, pack_stereo_script, "unpack",
                left_stereo_mp4, left_stereo_images_dir, left_mapping_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Count extracted images
                jpg_files = [f for f in os.listdir(left_stereo_images_dir) if f.endswith('.jpg')]
                print(f"✅ Unpacked {len(jpg_files)} left stereo images to {left_stereo_images_dir}")
                success = True
            else:
                print(f"❌ Error unpacking left stereo: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running pack_stereo.py for left stereo: {e}")
    else:
        print(f"⚠️  No left stereo MP4 or mapping found for {sequence_key}")
    
    # Unpack right stereo video
    if os.path.exists(right_stereo_mp4) and os.path.exists(right_mapping_file):
        right_stereo_images_dir = os.path.join(sequence_dir, "right_stereo_images")
        os.makedirs(right_stereo_images_dir, exist_ok=True)
        
        try:
            cmd = [
                sys.executable, pack_stereo_script, "unpack",
                right_stereo_mp4, right_stereo_images_dir, right_mapping_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Count extracted images
                jpg_files = [f for f in os.listdir(right_stereo_images_dir) if f.endswith('.jpg')]
                print(f"✅ Unpacked {len(jpg_files)} right stereo images to {right_stereo_images_dir}")
                success = True
            else:
                print(f"❌ Error unpacking right stereo: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error running pack_stereo.py for right stereo: {e}")
    else:
        print(f"⚠️  No right stereo MP4 or mapping found for {sequence_key}")
    
    return success

def print_summary(data_dir, sequences):
    """Print a summary of downloaded data"""
    print(f"\n📊 Download Summary")
    print(f"=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Sequences found: {len(sequences)}")
    
    total_size_mb = 0
    for sequence_key, files in sequences.items():
        print(f"\n  📁 {sequence_key}:")
        sequence_size = 0
        
        # Get the actual sequence directory path
        split_dir, sequence_id = sequence_key.split('/', 1)
        sequence_dir = os.path.join(data_dir, split_dir, sequence_id)
        
        for file in files:
            file_path = os.path.join(sequence_dir, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                sequence_size += size_mb
                
                # Categorize files
                if file.endswith('.mp4') and 'stereo' not in file and 'gt_view' not in file:
                    print(f"    🎥 User video: {size_mb:.1f} MB")
                elif file.endswith('.gt_view.mp4'):
                    print(f"    🎬 GT view video: {size_mb:.1f} MB")
                elif file.endswith('.left_stereo.mp4'):
                    print(f"    📹 Left stereo video: {size_mb:.1f} MB")
                elif file.endswith('.right_stereo.mp4'):
                    print(f"    📹 Right stereo video: {size_mb:.1f} MB")
                elif file.endswith('.npy'):
                    if "user_camera" in file:
                        file_type = "user camera intrinsics"
                    elif "gt_camera" in file:
                        file_type = "GT camera intrinsics"
                    elif "relative_transform" in file:
                        file_type = "relative transformation"
                    else:
                        file_type = "data"
                    print(f"    📐 {file_type}: {size_mb:.1f} MB")
                elif file.endswith('.csv'):
                    print(f"    📊 IMU data: {size_mb:.1f} MB")
                elif file.endswith('.h5'):
                    if "depth" in file:
                        print(f"    🏔️  Depth data: {size_mb:.1f} MB")
                    elif "confidence" in file:
                        print(f"    📊 Depth confidence: {size_mb:.1f} MB")
                    else:
                        print(f"    📊 H5 data: {size_mb:.1f} MB")
                elif file.endswith('.txt'):
                    if "gt_trajectory" in file:
                        print(f"    🛤️  GT trajectory: {size_mb:.1f} MB")
                    else:
                        print(f"    📄 Text data: {size_mb:.1f} MB")
                elif file.endswith('.json') and 'mapping' not in file:
                    print(f"    📋 Metadata: {size_mb:.1f} MB")
        
        print(f"    💾 Total: {sequence_size:.1f} MB")
        total_size_mb += sequence_size
    
    print(f"\n💾 Total dataset size: {total_size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Princeton365 dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract everything (default behavior)
  python download_princeton365.py --unpack-stereo
  
  # Download single sequence from validation split
  python download_princeton365.py --split validation --sample 0
  
  # Download in webdataset format (tar files)
  python download_princeton365.py --no-extract
  
  # Custom output directory
  python download_princeton365.py --output-dir ./my_data
        """
    )
    
    parser.add_argument("--repo-id", default="pvl-lab/princeton365-updated-webdataset",
                      help="HuggingFace repository ID (default: pvl-lab/princeton365-updated-webdataset)")
    
    parser.add_argument("--split", default="all",
                      help="Dataset split to download (default: all - downloads all available splits)")
    
    parser.add_argument("--sample", type=int, metavar="N",
                      help="Download only shard N (0-indexed). If not specified, downloads all shards")
    
    parser.add_argument("--no-extract", action="store_true",
                      help="Keep webdataset format (copy tar files to output directory without extracting)")
    
    parser.add_argument("--unpack-stereo", action="store_true",
                      help="Unpack stereo MP4 videos back to individual JPG images")
    
    parser.add_argument("--output-dir", default="princeton365_data",
                      help="Output directory for extracted data (default: princeton365_data)")
    
    parser.add_argument("--cache-dir", default=None,
                      help="Cache directory for HuggingFace downloads")
    
    parser.add_argument("--list-shards", action="store_true",
                      help="List available shards and exit")
    
    args = parser.parse_args()
    
    print("🎯 Princeton365 Dataset Downloader")
    print("=" * 50)
    
    # Determine splits to download from
    if args.split == "all":
        print(f"📋 Discovering available splits...")
        splits_to_download = discover_available_splits(args.repo_id)
        if not splits_to_download:
            print(f"❌ No splits found in {args.repo_id}")
            return
        print(f"📥 Found splits: {splits_to_download}")
    else:
        splits_to_download = [args.split]
    
    # List shards if requested
    if args.list_shards:
        for split in splits_to_download:
            print(f"📋 Listing available shards in {args.repo_id}/{split}...")
            shards = list_available_shards(args.repo_id, split)
            if shards:
                print(f"Split '{split}' available shards: {shards}")
            else:
                print(f"No shards found in split '{split}'")
        return
    
    # Download from all specified splits
    downloaded_files = []
    total_sequences = 0
    
    for split in splits_to_download:
        print(f"\n📥 Processing split: {split}")
        
        # Determine which shards to download from this split
        if args.sample is not None:
            shard_indices = [args.sample]
            print(f"📥 Downloading sample shard {args.sample} from {split} split...")
        else:
            print(f"📋 Discovering available shards in {split}...")
            shard_indices = list_available_shards(args.repo_id, split)
            if not shard_indices:
                print(f"❌ No shards found in {args.repo_id}/{split}")
                continue
            print(f"📥 Downloading {len(shard_indices)} shards from {split} split...")
        
        # Download shards from this split
        for shard_idx in shard_indices:
            downloaded_file = download_shard(args.repo_id, shard_idx, split, args.cache_dir)
            if downloaded_file:
                downloaded_files.append((downloaded_file, split))
                total_sequences += 1
    
    if not downloaded_files:
        print("❌ No files downloaded successfully")
        return
    
    print(f"✅ Downloaded {len(downloaded_files)} shard(s) from {len(splits_to_download)} split(s)")
    
    # Always copy files to output directory, either as webdataset format or extracted
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.no_extract:
        # Copy tar files to output directory in webdataset format
        print(f"\n📂 Copying webdataset to {args.output_dir}...")
        
        total_size_mb = 0
        for tar_path, split in downloaded_files:
            split_dir = os.path.join(args.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Copy tar file to split directory with original name
            tar_filename = os.path.basename(tar_path)
            dest_path = os.path.join(split_dir, tar_filename)
            shutil.copy2(tar_path, dest_path)
            
            size_mb = os.path.getsize(dest_path) / (1024*1024)
            total_size_mb += size_mb
            print(f"  📁 {tar_filename}: {size_mb:.1f} MB → {split_dir}")
        
        print(f"✅ Copied {len(downloaded_files)} tar files ({total_size_mb:.1f} MB total)")
        print(f"\n💡 WebDataset format preserved. Use with webdataset loaders:")
        
        # Show data_files structure for all splits
        splits_dict = {}
        for _, split in downloaded_files:
            if split not in splits_dict:
                splits_dict[split] = f"'{args.output_dir}/{split}/*.tar'"
        
        data_files_str = ', '.join([f"'{split}': {path}" for split, path in splits_dict.items()])
        print(f"   data_files = {{{data_files_str}}}")
        
    else:
        # Extract files to organized directory structure
        print(f"\n📦 Extracting to {args.output_dir}...")
        
        for tar_path, split in downloaded_files:
            extract_shard(tar_path, args.output_dir, split)
        
        # Find sequences in extracted data
        sequences = find_sequences(args.output_dir)
        
        if sequences:
            print_summary(args.output_dir, sequences)
            
            # Unpack stereo if requested
            if args.unpack_stereo:
                print(f"\n🎬 Unpacking stereo videos...")
                
                for sequence_key in sequences.keys():
                    unpack_stereo_video(args.output_dir, sequence_key)
        else:
            print("⚠️  No sequences found in extracted data")
    
    print(f"\n🎉 Download complete!")
    
    if args.no_extract:
        print(f"\n📁 WebDataset ready in: {args.output_dir}")
        print(f"   Structure: {args.output_dir}/{args.split}/*.tar")
        
        print(f"\n💡 Next steps:")
        print(f"   • Use with datasets: load_dataset('webdataset', data_files={{'{args.split}': '{args.output_dir}/{args.split}/*.tar'}})")
        print(f"   • Extract later: python download_princeton365.py --sample N (specific shard)")
        print(f"   • List contents: tar -tf {args.output_dir}/{args.split}/000000.tar")
        
    else:
        print(f"\n📁 Your data is ready in: {args.output_dir}")
        print(f"   Structure: {args.output_dir}/{args.split}/sequence_id/files")
        if args.unpack_stereo:
            print(f"📸 Left stereo images are in: {args.output_dir}/{args.split}/sequence_id/left_stereo_images/")
            print(f"📸 Right stereo images are in: {args.output_dir}/{args.split}/sequence_id/right_stereo_images/")


if __name__ == "__main__":
    main() 