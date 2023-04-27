# -*- coding: utf-8 -*-
"""
@author: dongs
version 0.4
change function so that it support both local picture read as well as the
camera capture system.
"""
import cv2
import numpy as np

def angle_between_points(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def process_frame(frame):
    gesture = "unknown"

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

        # Convex hull and convexity defects
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)

        # Find the bounding rectangle around the hand contour
        x, y, w, h = cv2.boundingRect(hand_contour)

        # Calculate the aspect ratio of the bounding rectangle
        aspect_ratio = float(w) / h

        # Gesture recognition
        if defects is not None:
            angles = []

            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                # Calculate the angle formed by the fingers
                angle = angle_between_points(np.array(start), np.array(far), np.array(end))
                angles.append(angle)

            # Find the average angle between the fingers
            avg_angle = np.mean(angles)

            if aspect_ratio > 1.0:
                gesture = "Clenched Fist"
            else:
                if avg_angle < 143:
                    gesture = "Victory Gesture"
                else:
                    gesture = "Index Finger Extended"

            # Display the gesture, aspect_ratio, and average angle on the frame
            cv2.putText(frame, f"Gesture: {gesture}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Aspect Ratio: {aspect_ratio:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Average Angle: {avg_angle:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def main():
    # Get the camera window size
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Unable to capture frame from camera")
        return

    window_size = (frame.shape[1], frame.shape[0])

    # Ask the user for the input method
    user_input = input("Enter 'c' for camera or 'l' for local image: ")

    if user_input.lower() == 'c':
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)

            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Exit on pressing the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    elif user_input.lower() == 'l':
        image_path = input("Enter the path to the image: ")
        frame = cv2.imread(image_path)
        
        if frame is None:
            print("Error: Image not found")
            return

        # Resize the local image to match the camera window size
        frame = cv2.resize(frame, window_size)

        frame = process_frame(frame)

        # Show the frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit on pressing any key
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid input. Please enter 'c' for camera or 'l' for local image.")

if __name__ == "__main__":
    main()

                    

                    