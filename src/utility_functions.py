import os
import numpy as np

def read_images_from_directory(directory: str) -> list:
    """
    Reads all the images from the specified directory and returns a list of image file paths.
    
    Parameters:
        directory (str): The path to the directory containing the images.
    
    Returns:
        list: A list of image file paths.
    """
    # List to store the image file paths
    images = []
    for filename in os.listdir(directory):
        # Check if the file is a PNG or JPG image
        if filename.endswith(".png") or filename.endswith(".jpg"):
            images.append(os.path.join(directory, filename))
    return images

def calculate_confidence_level(scores: np.ndarray, best_max_val: float) -> float:
    """
    This function calculates the confidence level of the best match found in the image
    based on the scores of the matches and the best match value. The confidence level
    is calculated as the ratio of the best match value to the 90th percentile of the scores.
    
    Parameters:
        scores (np.ndarray): The scores of the matches found in the image.
        best_max_val (float): The best match value found in the image.
    
    Returns:
        float: The confidence level of the best match found in the image.
    """
    # Calculate the 90th percentile of the scores
    percentile_score = np.percentile(scores, 90)
    
    # Handle the case where the 90th percentile score is zero
    if percentile_score == 0:
        confidence_level = float('inf') if best_max_val > 0 else 0.0
    else:
        # Calculate the confidence level
        confidence_level = best_max_val / percentile_score
    
    return confidence_level