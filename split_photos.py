import os
import shutil

# Define the path to the main data folder and the destination dataset folder
data_folder = 'data'
dataset_folder = 'dataset'

# Define the subfolders based on the prefixes
subfolders = ['bothFireAndSmoke', 'fire', 'neitherFireNorSmoke', 'smoke']

# Create the dataset folder and the subfolders if they don't exist
os.makedirs(dataset_folder, exist_ok=True)
for subfolder in subfolders:
    os.makedirs(os.path.join(dataset_folder, subfolder), exist_ok=True)

# Iterate through each file in the data folder
for filename in os.listdir(data_folder):
    if filename.startswith('bothFireAndSmoke_'):
        shutil.move(os.path.join(data_folder, filename), os.path.join(dataset_folder, 'bothFireAndSmoke', filename))
    elif filename.startswith('fire_'):
        shutil.move(os.path.join(data_folder, filename), os.path.join(dataset_folder, 'fire', filename))
    elif filename.startswith('neitherFireNorSmoke_'):
        shutil.move(os.path.join(data_folder, filename), os.path.join(dataset_folder, 'neitherFireNorSmoke', filename))
    elif filename.startswith('smoke_'):
        shutil.move(os.path.join(data_folder, filename), os.path.join(dataset_folder, 'smoke', filename))

print("Photos have been successfully split into their respective folders.")
