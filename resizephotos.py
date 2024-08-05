import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct full file path
        file_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
            try:
                with Image.open(file_path) as img:
                    # Resize image
                    img_resized = img.resize(size, Image.LANCZOS)
                    
                    # Save resized image to the output folder
                    img_resized.save(os.path.join(output_folder, filename))
                    
                print(f"Resized and saved {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Parameters
input_folder = r".\selected photos\input"
output_folder = r'.\selected photos\output'
size = (224, 224)  # Specify the desired size (width, height)

# Call the function
resize_images(input_folder, output_folder, size)
