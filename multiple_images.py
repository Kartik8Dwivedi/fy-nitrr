import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # Check if images are loaded correctly
    if img1 is None:
        print(f"Error: Image at path {image_path1} could not be loaded.")
    if img2 is None:
        print(f"Error: Image at path {image_path2} could not be loaded.")
    
    return img1, img2

def sobel_edge_detection(gray_img):
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.hypot(sobelx, sobely)
    _, edges_thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return edges_thresh.astype(np.uint8)

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_paper_transform(contour_img1, contour_img2):
    rect1 = cv2.minAreaRect(contour_img1)
    rect2 = cv2.minAreaRect(contour_img2)
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    M = cv2.getAffineTransform(np.float32(box1[:3]), np.float32(box2[:3]))
    return M, box1, box2

def plot_paper_outlines(img1_contours, img2_contours, M_paper1, M_paper2):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Image 1 with contours
    axes[0].imshow(cv2.cvtColor(img1_contours, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1: Red (Paper 1), Blue (Paper 2)', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # Add label for Image 1
    plt.text(10, 30, 'Image 1', fontsize=16, color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Image 2 with contours
    axes[1].imshow(cv2.cvtColor(img2_contours, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2: Red (Paper 1), Blue (Paper 2)', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # Add label for Image 2
    plt.text(10, 30, 'Image 2', fontsize=16, color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Display transformation matrices on the right side
    transformation_info = (
        f"Transformation Matrix for Paper 1:\n{np.array_str(M_paper1, precision=3, suppress_small=True)}\n\n"
        f"Transformation Matrix for Paper 2:\n{np.array_str(M_paper2, precision=3, suppress_small=True)}"
    )
    
    plt.figtext(0.75, 0.5, transformation_info, ha='center', fontsize=12, wrap=True, bbox=dict(facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()

def main(image_path1, image_path2):
    img1, img2 = load_images(image_path1, image_path2)
    
    # Ensure both images are loaded
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded. Exiting...")
        return
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Sobel edge detection
    edges1 = sobel_edge_detection(gray1)
    edges2 = sobel_edge_detection(gray2)

    # Find contours
    contours1 = find_contours(edges1)
    contours2 = find_contours(edges2)

    contour_paper1_img1 = max(contours1, key=cv2.contourArea)
    contour_paper2_img2 = max(contours2, key=cv2.contourArea)
    contour_paper2_img1 = sorted(contours1, key=cv2.contourArea)[-2]
    contour_paper2_img2 = sorted(contours2, key=cv2.contourArea)[-2]

    M_paper1, box1_paper1, box2_paper1 = get_paper_transform(contour_paper1_img1, contour_paper2_img2)
    M_paper2, box1_paper2, box2_paper2 = get_paper_transform(contour_paper2_img1, contour_paper2_img2)

    img1_contours = img1.copy()
    img2_contours = img2.copy()
    cv2.drawContours(img1_contours, [contour_paper1_img1], -1, (0, 0, 255), 3) 
    cv2.drawContours(img2_contours, [contour_paper2_img2], -1, (255, 0, 0), 3)  

    plot_paper_outlines(img1_contours, img2_contours, M_paper1, M_paper2)

image_path1 = './images/16.jpg'  
image_path2 = './images/17.jpg'  
main(image_path1, image_path2)
