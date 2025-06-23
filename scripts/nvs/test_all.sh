

#!/bin/bash

parent=/path/to/outputs

# Check if a directory is provided
if [ -z "$parent" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get absolute path of the directory
DIR=$(realpath "$parent")

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' not found!"
    exit 1
fi

## Find directories that contain a "config.yml" file directly under them
#find "$DIR" -type d -exec sh -c '[ -f "$1/config.yml" ] && echo "$1"' _ {} \;

# Loop through all subdirectories
# Find all directories containing a "config.yml" file
dirs=()
for config_file in $(find "$DIR" -type f -name "config.yml"); do
    dir=$(dirname "$config_file")
    dirs+=("$dir")
done

# Print the directories
for dir in "${dirs[@]}"; do
    echo "$dir"
    mkdir "$dir/eval"
    ns-eval --load-config "$dir/config.yml" --output-path "$dir/eval/metrics.json" --render-output-path "$dir/eval"
done