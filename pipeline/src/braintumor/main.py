import numpy as np
import os
import torchvision.transforms as transforms
from argparse import Namespace
from torch import Tensor
from PIL import Image

def get_data(args: Namespace) -> tuple[np.ndarray, np.ndarray, Namespace]:
    """
    Load and preprocess brain tumor image data.

    Args:
        args (Namespace): Arguments:
            - img_size (int): Size to which images will be resized.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Preprocessed feature data (images).
            - y (np.ndarray): Labels (0 for no tumor, 1 for tumor).
            - args (Namespace): Updated arguments with additional attributes:
                - features (tuple): Start and end dimensions of features.
                - n_features (int): Number of features.
                - n_classes (int): Number of classes.
    """
    X, y = _get_braintumor_data(args)
    args.features = (1, 3) # start dim, end dim
    args.n_features = X.reshape(X.shape[0], -1)[1]
    args.n_classes = 2
    return X, y, args
    
def _get_braintumor_data(args: Namespace):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..', '..', 'data', 'brain_tumor_dataset'))
    X, y = load_images(data_path, image_size=args.img_size)
    return X, y

def normalize_data(X_tensor: Tensor, args: Namespace) -> Tensor:
    return X_tensor # No further normalization applied


def load_images(data_folder: str, image_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses JPG images from subfolders 'yes' and 'no'.
    
    The subfolder names indicate the class:
      - "no": images without a brain tumor (label 0)
      - "yes": images with a brain tumor (label 1)
    
    Each image is resized, converted to a tensor, and then flattened.
    
    Args:
        data_folder (str): Path to the parent folder containing "yes" and "no" subfolders.
        image_size (int): Desired size (width, height) for image resizing.
        
    Returns:
        X (np.ndarray): Array of flattened image tensors.
        y (np.ndarray): Array of integer labels.
    """
    X = []
    y = []
    
    label_mapping = {"no": 0, "yes": 1}
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()  # Output tensor shape: (C, H, W)
    ])
    
    for class_name, label in label_mapping.items():
        class_folder = os.path.join(data_folder, class_name)
        if not os.path.isdir(class_folder):
            print(f"Warning: Folder '{class_folder}' not found. Skipping this class.")
            continue
        
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(class_folder, filename)
                try:
                    with Image.open(file_path) as img:
                        image = img.convert("RGB") # Ensure RGB
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                
                image_tensor = transform(image)
                X.append(image_tensor.numpy())
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    return X, y