import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from .utility_functions import calculate_confidence_level
from .image_processing_functions import adaptive_canny_edge_detection, non_max_suppression

def find_best_template_match_single_detection(image: np.ndarray, template: np.ndarray, scales: np.ndarray) -> tuple:
    """
    Find the best template match in the image for a range of scales when only one detection is expected.
    
    Parameters:
        image (np.ndarray): The image where the template is to be found.
        template (np.ndarray): The template to be found in the image.
        scales (np.ndarray): The scales to resize the template to.
    
    Returns:
        tuple: The best match, the best scale, the best maximum value, the top left corner of the best match, and the scores.
    """
    # Initialize the best match, scale, maximum value, and top left corner.
    best_match = None
    best_scale = None
    best_max_val = -np.inf
    best_top_left = None
    scores = []

    for scale in scales:
        # Resize the template to the current scale.
        resized_template = cv.resize(template, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        
        # Apply Canny edge detection to the resized template.
        resized_template = cv.Canny(resized_template, 10, 150)
        
        # Get the width and height of the resized template.
        w, h = resized_template.shape[::-1]

        if w <= image.shape[1] and h <= image.shape[0]:
            # Apply template matching to the image.
            res = cv.matchTemplate(image, resized_template, cv.TM_CCOEFF_NORMED)
            # Get the maximum value and its location.
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # Append the maximum value to the scores list.
            scores.append(max_val)
            
            # If the maximum value is greater than the best maximum value, 
            # update the best maximum value, scale, match, and top left corner.
            if max_val > best_max_val:
                best_max_val = max_val
                best_scale = scale
                best_match = resized_template
                best_top_left = max_loc

    return best_match, best_scale, best_max_val, best_top_left, scores

def detect_single_logo_in_image(img_path: str, template_path: str):
    """
    This function detects a single logo in an image.
    
    Parameters:
        image (str): The path to the image.
        template_path (str): The path to the template image
    
    Returns:
        None
    """
    # Load the main image and the template image
    image = cv.imread(img_path)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_path, 0)
    
    # Apply blur to the main image
    gray_image = cv.GaussianBlur(gray_image, (3, 3), 0)

    # Apply automatic Canny edge detection to the main image
    gray_image = adaptive_canny_edge_detection(gray_image)
    
    # Calculate the size ratio of the template to the main image
    height_ratio = template.shape[0] / gray_image.shape[0]
    width_ratio = template.shape[1] / gray_image.shape[1]
    
    # Set default min and max scales
    min_scale = 0.5
    max_scale = 1.5
    
    # If height ratio is too small, and the width ratio large
    if height_ratio < 0.3 and width_ratio > 0.5:
        min_scale = 0.2
    
    # If height and width ratios are too small
    if height_ratio < 0.2 and width_ratio < 0.3:
        min_scale = 3.0
        max_scale = 4.0
    
    print(f"Template height ratio: {height_ratio}")
    print(f"Template width ratio: {width_ratio}")

    # Define the scales to resize the template
    scales = np.linspace(min_scale, max_scale, 20)  

    # Find the best template match in the image for a range of scales
    best_match, best_scale, best_max_val, best_top_left, scores = find_best_template_match_single_detection(gray_image, template, scales)

    # Draw the bounding box on the original image
    if best_match is not None:
        bottom_right = (best_top_left[0] + best_match.shape[1], best_top_left[1] + best_match.shape[0])
        score = best_max_val
        top_left = best_top_left        
        
        confidence_level = calculate_confidence_level(scores, best_max_val)
        
        # Calculate the stroke thickness based on image dimensions
        stroke_thickness = max(1, int(min(image.shape[:2]) / 100))
        
        # Draw a rectangle around the best match
        cv.rectangle(image, best_top_left, bottom_right, (0, 255, 0), stroke_thickness)
        
        # Put the score text above the bounding box
        text = f'C: {confidence_level:.2f}'
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = min(image.shape[0], image.shape[1]) / 500
        font_thickness = 2
        text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = top_left[0]
        text_y = top_left[1] - 10 if top_left[1] - 10 > 10 else top_left[1] + text_size[1] + 10
        cv.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv.LINE_AA)

        # Print the results
        print(f"Best scale: {best_scale}")
        print(f"Best matching score: {best_max_val}")

        # Visualize the best matching result with the bounding box
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(f'Best Template Size (Scale: {best_scale})')
        
        
        plt.show()

def find_best_template_scale(image: np.ndarray, template: np.ndarray, scales: np.ndarray) -> float:
    """
    Find the best scale for the template based on the maximum matching score.
    
    Parameters:
        image (np.ndarray): The image where the template is to be found.
        template (np.ndarray): The template to be found in the image.
        scales (np.ndarray): The scales to resize the template to.
    
    Returns:
        best_scale (float): The best scale for the template.
    """
    # Initialize the best match, scale, maximum value, and top left corner.
    best_scale = None
    best_max_val = -np.inf

    for scale in scales:
        # Resize the template to the current scale.
        resized_template = cv.resize(template, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        
        # Apply Canny edge detection to the resized template.
        resized_template = cv.Canny(resized_template, 10, 150)
        
        # Get the width and height of the resized template.
        w, h = resized_template.shape[::-1]

        if w <= image.shape[1] and h <= image.shape[0]:
            # Apply template matching to the image.
            res = cv.matchTemplate(image, resized_template, cv.TM_CCOEFF_NORMED)
            # Get the maximum value and its location.
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            
            # If the maximum value is greater than the best maximum value, 
            # update the best maximum value, scale, match, and top left corner.
            if max_val > best_max_val:
                best_max_val = max_val
                best_scale = scale

    return best_scale

def detect_multiple_logo_in_image(img_path: str, template_path: str):
    """
    This function detects multiple instances of a logo in an image.
    
    Parameters:
        img_path (str): The path to the image.
        template_path (str): The path to the template.
    
    Returns:
        None
    """
    # Load the image and convert to grayscale
    img_rgb = cv.imread(img_path)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Load the template and get its dimensions
    template = cv.imread(template_path, 0)
    
    # Apply Gaussian blur to the image
    img_gray_blurred = cv.GaussianBlur(img_gray, (3, 3), 0)

    # Apply adaptive Canny edge detection to the image
    img_gray_edges = adaptive_canny_edge_detection(img_gray_blurred)

    # Calculate the size ratio of the template to the main image
    height_ratio = template.shape[0] / img_gray_blurred.shape[0]
    width_ratio = template.shape[1] / img_gray_blurred.shape[1]
    
    # Set default min and max scales
    min_scale = 0.5
    max_scale = 1.5
    
    # If height ratio is too small, and the width ratio large
    if height_ratio < 0.3 and width_ratio > 0.5:
        min_scale = 0.2
    
    # If height and width ratios are too small
    if height_ratio < 0.2 and width_ratio < 0.3:
        min_scale = 3.0
        max_scale = 4.0
    
    print(f"Template height ratio: {height_ratio}")
    print(f"Template width ratio: {width_ratio}")

    # Define the scales to resize the template
    scales = np.linspace(min_scale, max_scale, 20) 
    
    # Find the best scale for the template
    scale = find_best_template_scale(img_gray_edges, template, scales)

    print(f"Best scale: {scale}")

    # Resize the template to the best scale
    template = cv.resize(template, None, fx=scale, fy=scale)

    # Get the width and height of the template
    template_width, template_height = template.shape[::-1]

    # Apply Canny edge detection to the template
    template_edges = cv.Canny(template, 10, 150, L2gradient=True)

    # Apply template matching to the image
    res = cv.matchTemplate(img_gray_edges, template_edges, cv.TM_CCOEFF_NORMED)

    # Define a threshold to consider a match
    threshold = np.max(res) - 0.06

    # Find the locations where the matching score is greater than the threshold
    loc = np.where(res >= threshold)

    # Initialize the bounding boxes and individual scores
    boxes = []
    individual_scores = []

    for pt in zip(*loc[::-1]):
        # Add the bounding box and individual score to the lists
        boxes.append([pt[0], pt[1], template_width, template_height])
        individual_scores.append(res[pt[1], pt[0]])

    boxes = np.array(boxes)
    individual_scores = np.array(individual_scores)

    # Apply non-maximum suppression to the bounding boxes
    iou_threshold = 0.3
    keep, keep_scores = non_max_suppression(boxes, individual_scores, iou_threshold)

    # Compute the confidence level for each bounding box
    confidence_levels = [calculate_confidence_level(individual_scores, score) for score in keep_scores]

    # Get the global confidence level
    global_confidence_level = np.mean(confidence_levels)

    # Draw the final bounding boxes
    for i in keep:
        box = boxes[i]
        cv.rectangle(img_rgb, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

    # Display the image with the bounding boxes
    plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
    plt.title(f'Global Confidence Level: {global_confidence_level:.2f}')
    plt.show()