import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import math

# Global variables to hold two images
image1 = None
image2 = None

def load_first_image():
    global image1
    file_path = filedialog.askopenfilename()
    if file_path:
        image1 = cv2.imread(file_path)
        display_image_on_label(image1, image_label_1)

def load_second_image():
    global image2
    file_path = filedialog.askopenfilename()
    if file_path:
        image2 = cv2.imread(file_path)
        display_image_on_label(image2, image_label_2)

def display_image_on_label(img, label):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Resize image for display
    max_size = 400, 300
    img_pil.thumbnail(max_size, Image.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    return edges

def find_largest_paper_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_contour = approx
    return largest_contour

def compute_transformation_matrix(points1, points2):
    """Compute the full 3x3 affine transformation matrix."""
    M, _ = cv2.estimateAffinePartial2D(points1, points2)
    # Convert to 3x3 matrix
    M = np.vstack([M, [0, 0, 1]])
    return M

def extract_transform_info(M):
    """Extract translation, rotation from the 3x3 matrix."""
    tx = M[0, 2]
    ty = M[1, 2]
    rotation_rad = math.atan2(M[1, 0], M[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    return tx, ty, rotation_deg

def highlight_paper(img, paper_contour, label_text, color, origin=None):
    cv2.drawContours(img, [paper_contour], -1, color, 4)
    x, y = paper_contour[0][0]
    cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Mark the origin if provided
    if origin is not None and origin.any():
        ox, oy = origin
        cv2.circle(img, (int(ox), int(oy)), 5, (255, 0, 0), -1)  # Blue dot for origin
        cv2.putText(img, "Origin", (int(ox) + 10, int(oy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def process_images():
    global image1, image2
    if image1 is None or image2 is None:
        transformation_label.config(text="Please load both images!")
        return

    # Process first image
    edges1 = preprocess_image(image1)
    paper1 = find_largest_paper_contour(edges1)
    
    # Process second image
    edges2 = preprocess_image(image2)
    paper2 = find_largest_paper_contour(edges2)
    
    if paper1 is None or paper2 is None:
        transformation_label.config(text="No paper detected in one or both images.")
        return

    # Convert contours to points
    points1 = np.float32([point[0] for point in paper1])
    points2 = np.float32([point[0] for point in paper2])

    # Compute the transformation matrix
    M = compute_transformation_matrix(points1, points2)
    
    # Extract translation and rotation
    tx, ty, rotation_deg = extract_transform_info(M)

    # Display the full transformation matrix
    matrix_text = f"Transformation Matrix:\n{M}\n\nTranslation:\nX: {tx:.2f}, Y: {ty:.2f}\nRotation: {rotation_deg:.2f}°"

    # Mark the origin (let's assume the top-left corner of the first paper is the origin)
    origin = points1[0]

    # Highlight the paper in both images
    highlight_paper(image1, paper1, "Paper", (0, 255, 0), origin)  # Green for first image
    highlight_paper(image2, paper2, "Paper", (255, 0, 255), origin)  # Pink for second image

    # Update the GUI with the processed images and transformation results
    display_image_on_label(image1, image_label_1)
    display_image_on_label(image2, image_label_2)
    transformation_label.config(text=matrix_text)

# Create the GUI
root = tk.Tk()
root.title("2D Object Movement Detection")

# Create labels for displaying the two images
image_label_1 = Label(root)
image_label_1.pack(side=tk.LEFT)

image_label_2 = Label(root)
image_label_2.pack(side=tk.RIGHT)

# Buttons to load two images
load_button_1 = tk.Button(root, text="Load First Image", command=load_first_image)
load_button_1.pack()

load_button_2 = tk.Button(root, text="Load Second Image", command=load_second_image)
load_button_2.pack()

# Button to process images and detect paper translation
process_button = tk.Button(root, text="Detect Movement", command=process_images)
process_button.pack()

# Label to display translation results
transformation_label = Label(root, text="", justify="left")
transformation_label.pack()

# Start the GUI loop
root.mainloop()
