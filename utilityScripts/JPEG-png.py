import os
from PIL import Image

def convert_jpeg_to_png(directory, output_directory=None):
    """
    Converts all JPEG images in a directory to PNG format.
    
    Args:
    directory (str): The directory containing JPEG images.
    output_directory (str): The directory to save PNG images. If None, uses the same directory.
    """
    # Check if output directory is None, then set it to the input directory
    if output_directory is None:
        output_directory = directory

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            # Construct full file path
            full_path = os.path.join(directory, filename)
            # Open the image
            with Image.open(full_path) as img:
                # Define the output path with PNG extension
                output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.png")
                # Save the image in PNG format
                img.save(output_path, "PNG")
                print(f"Converted {filename} to PNG and saved as {output_path}")

# Usage example
source_directory = './search_img_dir'
output_directory = './final_search_img_dir'  # Optional: specify if you want to save in a different directory

convert_jpeg_to_png(source_directory, output_directory)