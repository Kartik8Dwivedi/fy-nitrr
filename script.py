import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_path1, image_path2):
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    return img1, img2

def apply_sobel_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  
    sobel_combined = cv2.magnitude(sobelx, sobely)        
    sobel_combined = np.uint8(sobel_combined)             
    return sobel_combined

def detect_and_match_keypoints(img1, img2):
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return keypoints1, keypoints2, matches

def compute_transformation_matrix(keypoints1, keypoints2, matches):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return H

def extract_translation_rotation_transformation(H):
    tx, ty = H[0, 2], H[1, 2]
    rotation_radians = np.arctan2(H[1, 0], H[0, 0])
    rotation_degrees = np.degrees(rotation_radians)
    return tx, ty, rotation_degrees

def plot_images_with_axes(img1, img2, H):
    origin = (0, 0)
    
    axis_length = 100
    axis_points = np.array([
        [origin[0], origin[1]],           
        [origin[0] + axis_length, origin[1]],  
        [origin[0], origin[1] + axis_length] 
    ], dtype=np.float32)

    img1_with_axes = img1.copy()
    cv2.line(img1_with_axes, tuple(axis_points[0].astype(int)), tuple(axis_points[1].astype(int)), (0, 0, 255), 2) 
    cv2.line(img1_with_axes, tuple(axis_points[0].astype(int)), tuple(axis_points[2].astype(int)), (0, 255, 0), 2) 

    transformed_points = cv2.perspectiveTransform(np.array([axis_points]), H)[0]

    img2_with_axes = img2.copy()
    cv2.line(img2_with_axes, tuple(transformed_points[0].astype(int)), tuple(transformed_points[1].astype(int)), (0, 0, 255), 2) 
    cv2.line(img2_with_axes, tuple(transformed_points[0].astype(int)), tuple(transformed_points[2].astype(int)), (0, 255, 0), 2)  

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Image 1 with Original Axes (Top-Left Corner Origin)')
    plt.imshow(cv2.cvtColor(img1_with_axes, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image 2 with Transformed Axes (Top-Left Corner Origin)')
    plt.imshow(cv2.cvtColor(img2_with_axes, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

def main(image_path1, image_path2):
    img1, img2 = load_images(image_path1, image_path2)

    edges1 = apply_sobel_edge_detection(img1)
    edges2 = apply_sobel_edge_detection(img2)

    keypoints1, keypoints2, matches = detect_and_match_keypoints(edges1, edges2)

    H = compute_transformation_matrix(keypoints1, keypoints2, matches)

    tx, ty, rotation_degrees = extract_translation_rotation_transformation(H)

    print("Transformation Matrix:\n", H)
    print(f"Translation: x = {tx:.2f}, y = {ty:.2f}")
    print(f"Rotation: {rotation_degrees:.2f} degrees")

    plot_images_with_axes(img1, img2, H)

image_path1 = './images/12.jpg'
image_path2 = './images/11.jpg'
main(image_path1, image_path2)
