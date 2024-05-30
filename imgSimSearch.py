import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.io import read_image
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity

# Set up paths for image directories
image_dir = './final_search_img_dir'
new_image_path = './candidate_img_dir/n01440764_tench.png'

# Choose the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),               # Resize to 256 pixels along the smallest dimension
    transforms.CenterCrop(224),           # Crop a square of 224 x 224 pixels from center
    transforms.ConvertImageDtype(torch.float32),  # Convert image to floating point
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def load_and_preprocess_image(image_path):
    """Load and preprocess an image from a file."""
    image = read_image(image_path)
    if image.shape[0] == 1:  # If grayscale, replicate channels to make it RGB
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:  # If RGBA, remove the alpha channel
        image = image[:3, :, :]
    return preprocess(image).unsqueeze(0).to(device)

class FeatureExtractor(nn.Module):
    """A feature extractor that utilizes a pretrained VGG16 network."""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Exclude the classifier

    def forward(self, x):
        """Extract features from input batch."""
        with torch.no_grad():  # No need for gradients
            x = self.features(x)
            return torch.flatten(x, 1)  # Flatten the features to a vector

def save_embeddings_hdf5(embeddings, file_name='image_embeddings.h5'):
    """Save the image embeddings to an HDF5 file."""
    with h5py.File(file_name, 'w') as f:
        for img_name, feature in embeddings.items():
            f.create_dataset(img_name, data=feature.cpu().numpy())

def load_embeddings_hdf5(file_name='image_embeddings.h5'):
    """Load image embeddings from an HDF5 file."""
    embeddings = {}
    with h5py.File(file_name, 'r') as f:
        for img_name in f.keys():
            embeddings[img_name] = torch.tensor(f[img_name][:]).to(device)
    return embeddings

def compute_embeddings(directory, model):
    """Compute embeddings for all images in the specified directory."""
    embeddings = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        image = load_and_preprocess_image(img_path)
        embeddings[img_name] = model(image)
    return embeddings

def compare_new_image(image_path, embeddings, model):
    """Compare a new image with the precomputed embeddings database."""
    new_image = load_and_preprocess_image(image_path)
    new_features = model(new_image).view(1, -1).cpu().numpy()  # Flatten features for comparison
    similarities = {}
    for img_name, features in embeddings.items():
        features = features.view(1, -1).cpu().numpy()  # Flatten features for comparison
        sim = cosine_similarity(new_features, features)[0][0]  # Calculate cosine similarity
        similarities[img_name] = sim
    return similarities

# # Main execution flow
# model = FeatureExtractor()
# model.eval()  # Set model to evaluation mode (disables dropout and batch norm)

# embeddings = compute_embeddings(image_dir, model)  # Compute embeddings for all images
# save_embeddings_hdf5(embeddings)  # Save computed embeddings

# loaded_embeddings = load_embeddings_hdf5()  # Load saved embeddings
# similarities = compare_new_image(new_image_path, loaded_embeddings, model)  # Compare new image
# print(similarities)  # Print the similarities
