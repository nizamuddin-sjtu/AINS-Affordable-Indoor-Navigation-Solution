#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import math

def gaussian_filter(image):
    """
    Apply Gaussian filtering to remove noise from the image.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def convert_rgb_to_hsv(image):
    """
    Convert the RGB image to HSV color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def detect_yellow_color(hsv_image):
    """
    Detect the yellow color in the HSV image using predefined thresholds.
    """
    lower_yellow = np.array([25, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return mask

def find_contours(mask):
    """
    Find contours in the binary mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_central_moments(contour):
    """
    Calculate the central moments of the detected contour.
    """
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    return cx, cy

def calculate_angle(cx, cy):
    """
    Calculate the angle between the reference vector and the central moments of the path.
    """
    angle = math.atan2(cy, cx)
    angle_degrees = math.degrees(angle)
    return angle_degrees

def navigate_vehicle(angle):
    """
    Navigate the vehicle based on the calculated angle.
    """
    if angle < -10:
        print("Turn left")
    elif angle > 10:
        print("Turn right")
    else:
        print("Go straight")

# Main method
def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with a video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Preprocess the image
        blurred_frame = gaussian_filter(frame)

        # Step 2: Convert the image to HSV color space
        hsv_image = convert_rgb_to_hsv(blurred_frame)

        # Step 3: Detect the yellow color in the HSV image
        mask = detect_yellow_color(hsv_image)

        # Step 4: Find contours in the mask
        contours = find_contours(mask)
        
        if contours:
            # Assume the largest contour corresponds to the path
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Step 5: Calculate central moments of the largest contour
            cx, cy = calculate_central_moments(largest_contour)
            
            # Step 6: Calculate the angle based on central moments
            angle = calculate_angle(cx, cy)
            
            # Step 7: Navigate the vehicle based on the calculated angle
            navigate_vehicle(angle)
            
            # Draw the contour and center point on the original frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # Display the original frame
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




