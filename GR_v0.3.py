# -*- coding: utf-8 -*-
"""
@author: dongs
version 0.3
change algorithm
"""

import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    while True:
        gesture = "unknown"
        distance_ratio = 0.0
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (35, 35), 0)

        # Threshold the image to create a binary image
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours based on a minimum area threshold
        min_area = 5000
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if filtered_contours:
            hand_contour = max(filtered_contours, key=cv2.contourArea)
            M = cv2.moments(hand_contour)
            if M["m00"] != 0:
                center = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            else:
                center = np.array([0, 0])

            # Convex hull and convexity defects
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            defects = cv2.convexityDefects(hand_contour, hull)

            # Find the bounding rectangle around the hand contour
            x, y, w, h = cv2.boundingRect(hand_contour)

            # Calculate the aspect ratio of the bounding rectangle
            aspect_ratio = float(w) / h

            # Gesture recognition
            if defects is not None:
                far_points = []
                for i in range(defects.shape[0]):
                    s, e, f, _ = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])

                    # Calculate the distance between the farthest point and the center
                    far_distance = np.linalg.norm(center - np.array(far))
                    far_points.append((far, far_distance))

                # Sort the far points by distance
                far_points.sort(key=lambda x: x[1], reverse=True)

                if len(far_points) >= 2:
                    # Check the aspect ratio to differentiate between the "Clenched Fist" and other gestures
                    if aspect_ratio > 1.2:
                        gesture = "Clenched Fist"
                    else:
                        # Calculate the distance ratio between the two farthest points
                        distance_ratio = far_points[0][1] / far_points[1][1]

                        if distance_ratio > 1.5:
                            gesture = "Index Finger Extended"
                        else: 
                            gesture = "Victory Gesture"
                else:
                    gesture = "Clenched Fist"
            else:
                gesture = "unknown"

            # Draw the hand contour and the bounding rectangle
            cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the gesture, aspect_ratio, and distance_ratio on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Aspect Ratio: {aspect_ratio:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if defects is not None and len(far_points) >= 2:
                cv2.putText(frame, f"Distance Ratio: {distance_ratio:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            
        # Show the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
