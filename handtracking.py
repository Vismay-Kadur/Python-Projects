import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        # Region of Interest (ROI) for hand tracking
        self.roi_top = 100
        self.roi_bottom = 400
        self.roi_right = 350
        self.roi_left = 650

        # Background subtraction
        self.bg = None
        self.accumulated_weight = 0.5

        # Calibration
        self.calibrated = False
        self.num_frames = 0
        self.calibration_frames = 30

    def calibrate_background(self, frame):
        """Capture background for better hand detection"""
        roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.bg is None:
            self.bg = gray.copy().astype("float")
            return

        cv2.accumulateWeighted(gray, self.bg, self.accumulated_weight)

    def segment_hand(self, frame):
        """Segment hand from background using absolute difference"""
        roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Calculate absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), gray)

        # Threshold the difference
        _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresholded = cv2.morphologyEx(
            thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresholded = cv2.morphologyEx(
            thresholded, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None

        # Get the largest contour (hand)
        hand_contour = max(contours, key=cv2.contourArea)

        # Only return if contour is large enough
        if cv2.contourArea(hand_contour) > 2000:
            return thresholded, hand_contour

        return None, None

    def count_fingers(self, thresholded, hand_contour):
        """Count fingers using convex hull and defects"""
        # Find convex hull
        hull = cv2.convexHull(hand_contour, returnPoints=False)

        if len(hull) > 3:
            # Find convexity defects
            defects = cv2.convexityDefects(hand_contour, hull)

            if defects is not None:
                finger_count = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])

                    # Calculate distances
                    a = np.sqrt((end[0] - start[0])**2 +
                                (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 +
                                (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    # Calculate angle
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                    # If angle is less than 90 degrees, treat as finger
                    if angle <= np.pi/2:
                        finger_count += 1

                return finger_count + 1  # Add 1 for base

        return 0

    def draw_roi(self, frame):
        """Draw Region of Interest box"""
        cv2.rectangle(frame, (self.roi_right, self.roi_top),
                      (self.roi_left, self.roi_bottom), (0, 255, 0), 2)
        return frame


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector()

    print("=" * 60)
    print("HAND TRACKING - CALIBRATION MODE")
    print("=" * 60)
    print("\nINSTRUCTIONS:")
    print("1. Keep your hand OUT of the green box")
    print("2. Wait for 30 frames to calibrate background")
    print("3. Then place your hand INSIDE the green box")
    print("4. Press 'r' to recalibrate at any time")
    print("5. Press 'q' to quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        # Draw ROI
        frame = detector.draw_roi(frame)

        # Calibration phase
        if detector.num_frames < detector.calibration_frames:
            detector.calibrate_background(frame_copy)
            detector.num_frames += 1

            cv2.putText(frame, f"CALIBRATING... {detector.num_frames}/{detector.calibration_frames}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Keep hand OUT of box!",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            if not detector.calibrated:
                detector.calibrated = True
                print("\n✓ Calibration complete! Place hand in green box.")

            # Segment hand
            thresholded, hand_contour = detector.segment_hand(frame_copy)

            if thresholded is not None and hand_contour is not None:
                # Draw contour on ROI
                roi = frame[detector.roi_top:detector.roi_bottom,
                            detector.roi_right:detector.roi_left]

                cv2.drawContours(roi, [hand_contour], -1, (255, 0, 0), 2)

                # Count fingers
                finger_count = detector.count_fingers(
                    thresholded, hand_contour)

                # Display finger count
                cv2.putText(frame, f"Fingers: {finger_count}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Show thresholded image
                cv2.imshow("Hand Segmentation", thresholded)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display
        cv2.imshow("Hand Tracking", frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset calibration
            detector.bg = None
            detector.num_frames = 0
            detector.calibrated = False
            print("\n⟳ Recalibrating... Keep hand out of box!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
