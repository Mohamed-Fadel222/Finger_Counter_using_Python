import cv2
import numpy as np

def count_fingers(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get binary image
    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to find hand contour
    max_area = 0
    hand_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            hand_contour = contour

    # If a hand contour is found
    if hand_contour is not None:
        # Calculate the convex hull
        hull = cv2.convexHull(hand_contour)

        # Draw contours and hull
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)

        # Calculate the number of fingers using defects in the convex hull
        defects = cv2.convexityDefects(hand_contour, cv2.convexHull(hand_contour, returnPoints=False))
        finger_count = 0

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                # Calculate the triangle formed by defects
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                # If angle is less than 90 degrees, consider it as a finger
                if angle < np.pi/2:
                    finger_count += 1
                    cv2.circle(frame, far, 5, [0, 0, 255], -1)

        return frame, finger_count

    return frame, 0

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirrored view

    processed_frame, fingers = count_fingers(frame)

    cv2.putText(processed_frame, f"Finger Count: {fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Finger Counter', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
