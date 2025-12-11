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

        # Sign language history for stability
        self.sign_history = []
        self.history_size = 10

    def calibrate_background(self, frame):
        """Capture background for better hand detection""",
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

    def analyze_hand_features(self, thresholded, hand_contour):
        """Analyze hand features for sign language recognition"""
        # Calculate basic features
        area = cv2.contourArea(hand_contour)
        hull = cv2.convexHull(hand_contour)
        hull_area = cv2.contourArea(hull)

        # Solidity (how "filled" the hand shape is)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Aspect ratio
        x, y, w, h = cv2.boundingRect(hand_contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Count fingers
        finger_count = self.count_fingers_advanced(thresholded, hand_contour)

        # Perimeter
        perimeter = cv2.arcLength(hand_contour, True)

        # Extent (ratio of contour area to bounding box area)
        extent = float(area) / (w * h) if (w * h) > 0 else 0

        return {
            'fingers': finger_count,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'perimeter': perimeter,
            'extent': extent,
            'width': w,
            'height': h
        }

    def count_fingers_advanced(self, thresholded, hand_contour):
        """Advanced finger counting using convex hull and defects"""
        hull = cv2.convexHull(hand_contour, returnPoints=False)

        if len(hull) > 3 and len(hand_contour) > 3:
            try:
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
                        c = np.sqrt((end[0] - far[0])**2 +
                                    (end[1] - far[1])**2)

                        # Calculate angle
                        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))

                        # If angle is less than 90 degrees and depth is reasonable
                        if angle <= np.pi/2 and d > 5000:
                            finger_count += 1

                    return finger_count
            except:
                pass

        return 0

    def recognize_sign(self, features):
        """Recognize ASL sign based on hand features"""
        fingers = features['fingers']
        solidity = features['solidity']
        aspect_ratio = features['aspect_ratio']
        extent = features['extent']

        # Sign language recognition logic (simplified)
        sign = "?"

        # A - Fist (closed hand, high solidity)
        if fingers == 0 and solidity > 0.85:
            sign = "A"

        # B - Flat hand (all fingers up, high solidity)
        elif fingers >= 3 and solidity > 0.75 and aspect_ratio < 0.7:
            sign = "B"

        # C - Curved hand (medium solidity, curved shape)
        elif fingers == 0 and 0.6 < solidity < 0.8 and extent < 0.7:
            sign = "C"

        # D - Index finger up, others closed
        elif fingers == 1 and solidity > 0.7:
            sign = "D"

        # E - Closed fist (similar to A but slightly different)
        elif fingers == 0 and solidity > 0.88:
            sign = "E"

        # F - OK sign (low solidity due to hole)
        elif fingers == 0 and solidity < 0.7:
            sign = "F"

        # I - Pinky up
        elif fingers == 1 and aspect_ratio > 1.2:
            sign = "I"

        # L - L shape (thumb and index)
        elif fingers == 1 and 0.8 < aspect_ratio < 1.5:
            sign = "L"

        # O - Circle shape (low solidity)
        elif fingers == 0 and 0.5 < solidity < 0.7 and 0.7 < aspect_ratio < 1.3:
            sign = "O"

        # V - Two fingers (peace sign)
        elif fingers == 2:
            sign = "V"

        # W - Three fingers
        elif fingers == 3:
            sign = "W"

        # Y - Thumb and pinky extended
        elif fingers == 2 and aspect_ratio > 1.5:
            sign = "Y"

        # Five fingers extended
        elif fingers >= 4:
            sign = "5"

        return sign

    def get_stable_sign(self, sign):
        """Use history to stabilize sign recognition"""
        self.sign_history.append(sign)

        if len(self.sign_history) > self.history_size:
            self.sign_history.pop(0)

        # Return most common sign in history
        if len(self.sign_history) >= 5:
            from collections import Counter
            most_common = Counter(self.sign_history).most_common(1)
            return most_common[0][0]

        return sign

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
    print("ASL SIGN LANGUAGE RECOGNITION")
    print("=" * 60)
    print("\nINSTRUCTIONS:")
    print("1. Keep your hand OUT of the green box during calibration")
    print("2. Wait for calibration to complete (30 frames)")
    print("3. Place your hand INSIDE the green box")
    print("4. Make ASL signs - the letter will be displayed")
    print("\nSupported signs (simplified):")
    print("A, B, C, D, E, F, I, L, O, V, W, Y, and numbers")
    print("\nControls:")
    print("  'r' - Recalibrate")
    print("  'q' - Quit")
    print("=" * 60)

    current_word = ""
    last_sign = None
    sign_hold_frames = 0
    frames_to_add = 20  # Hold sign for 20 frames to add to word

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
                print("\n✓ Calibration complete! Start making signs.")

            # Segment hand
            thresholded, hand_contour = detector.segment_hand(frame_copy)

            if thresholded is not None and hand_contour is not None:
                # Draw contour on ROI
                roi = frame[detector.roi_top:detector.roi_bottom,
                            detector.roi_right:detector.roi_left]

                cv2.drawContours(roi, [hand_contour], -1, (255, 0, 0), 2)

                # Analyze hand features
                features = detector.analyze_hand_features(
                    thresholded, hand_contour)

                # Recognize sign
                sign = detector.recognize_sign(features)
                stable_sign = detector.get_stable_sign(sign)

                # Display recognized sign (large)
                cv2.putText(frame, f"Sign: {stable_sign}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                # Display finger count
                cv2.putText(frame, f"Fingers: {features['fingers']}",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Word building logic
                if stable_sign != "?" and stable_sign != last_sign:
                    sign_hold_frames = 1
                    last_sign = stable_sign
                elif stable_sign == last_sign and stable_sign != "?":
                    sign_hold_frames += 1
                    if sign_hold_frames == frames_to_add:
                        current_word += stable_sign
                        print(f"Added: {stable_sign} | Word: {current_word}")

                # Show thresholded image
                cv2.imshow("Hand Segmentation", thresholded)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                last_sign = None
                sign_hold_frames = 0

        # Display current word
        cv2.putText(frame, f"Word: {current_word}",
                    (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        cv2.putText(frame, "Press SPACE to clear word",
                    (10, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Display
        cv2.imshow("ASL Sign Language Recognition", frame)

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
        elif key == ord(' '):
            # Clear word
            current_word = ""
            print("\n✓ Word cleared!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
