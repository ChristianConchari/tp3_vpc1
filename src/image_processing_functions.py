import numpy as np
import cv2 as cv

def adaptive_canny_edge_detection(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Apply Canny edge detection using adaptive thresholding based on the median pixel intensity.
    
    The function calculates the thresholds for Canny edge detection based on the median 
    of the pixel intensities in the image. This method ensures that the thresholds are 
    automatically adjusted according to the contrast of the image.

    Parameters:
        image (np.ndarray): Input grayscale image.
        sigma (float): Standard deviation factor for adjusting the thresholds around the median value.

    Returns:
        np.ndarray: Edge-detected image.
    """
    # Adjust thresholds based on the median pixel intensity
    median_value = np.median(image)    
    lower = int(max(0, (1.0 - sigma) * median_value))
    upper = int(min(255, (1.0 + sigma) * median_value))
    
    # Adjust thresholds based on the contrast level of the image
    if median_value > 150:
        print("High contrast image")
        lower = 100
        upper = 540
        
    if 75 < median_value < 150:
        lower = 300
        upper = 450
    
    if median_value <= 75:
        lower = 10
        upper = 250
    
    # Apply Canny edge detection
    edged_image = cv.Canny(image, lower, upper, L2gradient=True)
    print(f"Median value: {median_value}, lower: {lower}, upper: {upper}")
    
    # Dilate the edges to make them more pronounced
    edged_image = cv.dilate(edged_image, None, iterations=1)
    
    return edged_image

def non_max_suppression(boxes: np.array, scores: np.array, iou_threshold: float) -> tuple:
    """
    Performs non-maximum suppression on a set of bounding boxes based 
    on their scores and intersection-over-union (IoU) threshold.
    
    Parameters:
        boxes (numpy.ndarray): An array of shape (N, 4) representing the coordinates of the top-left corner and the bottom-right corner of each box.
        scores (numpy.ndarray): An array of shape (N,) representing the scores of each box.
        iou_threshold (float): The threshold value for the IoU. Boxes with an IoU greater than this threshold will be suppressed.
        
    Returns:
        tuple: A tuple containing a list of indices representing the boxes to keep after non-maximum suppression and their corresponding scores.
    """
    # Extract the coordinates of the top-left corner and the bottom-right corner of each box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Calculate the area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort the boxes by their scores in descending order
    order = scores.argsort()[::-1]

    # Initialize the list of indices to keep
    keep = []
    keep_scores = []
    
    while order.size > 0:
        # Select the box with the highest score
        i = order[0]
        keep.append(i)
        keep_scores.append(scores[i])

        # Find the coordinates of the intersection area between the selected box and the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate the width and height of the intersection area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Calculate the area of the intersection
        inter = w * h
        
        # Calculate the Intersection-over-Union (IoU) for the intersection area
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep only the boxes with an IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        
        order = order[inds + 1]

    return keep, keep_scores