import os
import subprocess
import json
import glob

def images_to_mp4(input_dir, output_dir, output_name):
    """
    Convert timestamped JPG images to MP4 video.
    
    Args:
        input_dir: Directory containing JPG files named with timestamps
        output_dir: Output directory for MP4 file and frame mapping
    """

    os.makedirs(output_dir, exist_ok=True)
    # Get all jpg files and sort them
    jpg_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    jpg_files.sort()
    
    if not jpg_files:
        print("No JPG files found in directory")
        return
    
    # Create mapping file to preserve original filenames
    mapping = {}
    for i, filepath in enumerate(jpg_files):
        filename = os.path.basename(filepath)
        mapping[i + 1] = filename  # frame numbers start at 1
    
    mapping_file = os.path.join(output_dir, "frame_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f)
    
    # Run ffmpeg to create video
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite output file
        "-framerate", "60",
        "-pattern_type", "glob",
        "-i", os.path.join(input_dir, "*.jpg"),
        "-s", "1920x1080",
        "-c:v", "libx264",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        os.path.join(output_dir, output_name)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created: {os.path.join(output_dir, output_name)}")
        print(f"Mapping saved: {mapping_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")

def mp4_to_images(input_mp4, output_dir, mapping_file):
    """
    Convert MP4 video back to timestamped JPG images.
    
    Args:
        input_mp4: Input MP4 file path
        output_dir: Directory to save extracted images
        mapping_file: JSON file with frame-to-filename mapping
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames to temporary numbered files
    temp_pattern = os.path.join(output_dir, "temp_%06d.jpg")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-q:v", "2",  # High quality
        "-f", "image2",
        temp_pattern
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Frames extracted successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return
    
    # Load mapping and restore original filenames
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        for frame_num_str, original_filename in mapping.items():
            frame_num = int(frame_num_str)
            temp_filename = f"temp_{frame_num:06d}.jpg"
            temp_path = os.path.join(output_dir, temp_filename)
            original_path = os.path.join(output_dir, original_filename)
            
            if os.path.exists(temp_path):
                os.rename(temp_path, original_path)
        
        # Clean up any remaining temp files
        for temp_file in glob.glob(os.path.join(output_dir, "temp_*.jpg")):
            os.remove(temp_file)
        
        print(f"Images restored to original filenames in: {output_dir}")
        
    except FileNotFoundError:
        print(f"Mapping file not found: {mapping_file}")
    except Exception as e:
        print(f"Error restoring filenames: {e}")

# Command line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Pack images to MP4:")
        print("    python pack_stereo.py pack <input_images_dir> <output_dir> <output_mp4_name>")
        print("  Unpack MP4 to images:")
        print("    python pack_stereo.py unpack <input_mp4> <output_images_dir> <mapping_file>")
        print("\nExample:")
        print("  python pack_stereo.py pack stereo_images/ output/ sequence.mp4")
        print("  python pack_stereo.py unpack sequence.mp4 unpacked_images/ frame_mapping.json")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "pack":
        if len(sys.argv) != 5:
            print("Error: pack requires 3 arguments: <input_images_dir> <output_dir> <output_mp4_name>")
            sys.exit(1)
        
        input_dir, output_dir, mp4_name = sys.argv[2], sys.argv[3], sys.argv[4]
        print(f"Packing images from {input_dir} to {output_dir}/{mp4_name}")
        images_to_mp4(input_dir, output_dir, mp4_name)
        
    elif command == "unpack":
        if len(sys.argv) != 5:
            print("Error: unpack requires 3 arguments: <input_mp4> <output_images_dir> <mapping_file>")
            sys.exit(1)
        
        input_mp4, output_dir, mapping_file = sys.argv[2], sys.argv[3], sys.argv[4]
        print(f"Unpacking {input_mp4} to {output_dir} using {mapping_file}")
        mp4_to_images(input_mp4, output_dir, mapping_file)
        
    else:
        print(f"Error: Unknown command '{command}'. Use 'pack' or 'unpack'.")
        sys.exit(1)
    