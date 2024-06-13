import os
import shutil
import random

def rename_and_move_files(src_dir, dest_dir, class_name, label_info):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for idx, filename in enumerate(os.listdir(src_dir)):
        if filename.endswith(".wav") and not filename.startswith("._"):  # Skip hidden/system files
            # Replace underscores in the original filename with hyphens
            tape_name = os.path.splitext(filename)[0].replace('_', '-')
            # Generate random start and end times in milliseconds
            start_time_ms = f"{random.randint(100000, 999999)}"
            end_time_ms = f"{random.randint(100000, 999999)}"
            # Semi-randomize years to ensure proper partitioning
            year = random.choice([2020, 2021, 2022, 2023, 2024])

            # Construct the new filename according to the naming convention
            new_filename = f"{class_name}-{label_info}sec_{idx:04d}_{year}_{tape_name}_{start_time_ms}_{end_time_ms}.wav"

            # Source and destination paths
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, new_filename)

            # Move and rename the file
            shutil.copy(src_path, dest_path)
            print(f"Moved: {src_path} -> {dest_path}")

# Define the main directory for ANIMAL-SPOT data
animal_spot_data_dir = r"/home/gridsan/abradshaw/MITBuzz/ANIMAL-SPOT/ANIMAL-SPOT-DATA"

# Define the source directories for positive and negative datasets
positive_dataset_dir = r"/home/gridsan/abradshaw/MITBuzz/Positive_1sec_20240607"
negative_dataset_dir = r"/home/gridsan/abradshaw/MITBuzz/Negative_1sec_20240607"

# Define the target directories within the ANIMAL-SPOT data directory
target_dir = os.path.join(animal_spot_data_dir, "target")
noise_dir = os.path.join(animal_spot_data_dir, "noise")

# Rename and move positive files (target) to the target directory
rename_and_move_files(positive_dataset_dir, target_dir, "target", "positivedataset1")

# Rename and move negative files (noise) to the noise directory
rename_and_move_files(negative_dataset_dir, noise_dir, "noise", "negativedataset1")

print("All files have been moved and renamed successfully.")
