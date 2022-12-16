import os
import time
from pathlib import Path
from tqdm import tqdm

# The path to the directory with the jpg files
jpg_dir = Path("/home/hoenig/iChores/datasets/Objects365/images/train")

# The path to the directory with the txt files
txt_dir = Path("/home/hoenig/iChores/datasets/Objects365/labels/train")

# Calculate the total number of files that need to be processed
total_num_files = sum(1 for _ in jpg_dir.glob("*.jpg"))

# Estimate the number of files that can be processed per second
files_per_second = 100

# Calculate the estimated time it will take for the script to run
estimated_time = total_num_files / files_per_second

# Record the start time
start_time = time.time()

# Print the estimated time
print(f"Estimated time to complete: {estimated_time} seconds")

# Iterate over the jpg files
for jpg_file in tqdm(jpg_dir.glob("*.jpg"), total=total_num_files):
  # Get the file name without the extension
  jpg_name = jpg_file.stem

  # Check if there's a non-empty txt file with the same name
  txt_file = txt_dir / (jpg_name + ".txt")
  if not txt_file.is_file() or not txt_file.stat().st_size:
    # If there's no txt file with the same name, delete the jpg file
    jpg_file.unlink()

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time}")