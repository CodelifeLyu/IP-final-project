# -*- coding: utf-8 -*-
"""
@author: dongs
version 0.1
proposal version
"""

import cv2
import numpy as np

# Parameters
width, height = 300, 300
defect_threshold = 10000

# Initialize the camera
cap = cv2.VideoCapture(0)

# Background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=True)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Background subtraction
    fgmask = background_subtractor.apply(blurred)
    _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    # Contour extraction
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and solidity
    min_area = 1000
    min_solidity = 0.9
    
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
    
        if hull_area > 0:
            solidity = float(area) / hull_area
    
            if area > min_area and solidity > min_solidity:
                filtered_contours.append(cnt)

    if filtered_contours:
        hand_contour = max(filtered_contours, key=cv2.contourArea)
    
        # Convex hull and convexity defects
        convex_hull = cv2.convexHull(hand_contour)
        convexity_defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))
    
        defects = []
        if convexity_defects is not None:
            for i in range(convexity_defects.shape[0]):
                s, e, f, d = convexity_defects[i, 0]
                if d > defect_threshold:
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])
                    defects.append((start, end, far))
    
        # Gesture recognition
        if len(defects) == 0:
            gesture = "clenched_fist"
        elif len(defects) == 1:
            gesture = "index_finger_extended"
        elif len(defects) == 2:
            gesture = "victory_gesture"
        else:
            gesture = "unknown"
    else:
        gesture = "unknown"
    # Output
    print(f"Recognized gesture: {gesture}")

    # Display the original frame and thresholded frame
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Thresholded Frame', thresh)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
