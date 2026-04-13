import cv2
import time
from face_detection import FaceDetector
from fatigue_detection import FatigueDetector
from alerts import AlertSystem
from object_detection import ObjectDetector
from logger import SessionLogger

def main():
    print("Initializing Webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Initialize modules
    detector = FaceDetector()
    fatigue_system = FatigueDetector()
    alert_system = AlertSystem()
    obj_detector = ObjectDetector()
    session_logger = SessionLogger()

    print("Starting Fatigue Detection System. Press 'q' to quit.")
    prev_time = time.time()
    
    # Calibration variables
    CALIBRATION_STAGES = ["Center", "Left", "Right", "Down"]
    calib_idx = 0
    calib_start_time = time.time()
    STAGE_DURATION = 4.0
    
    is_calibrating = True
    
    # Accumulated data for calibration
    center_ear_list = []
    center_mar_list = []
    bounds = {}

    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame. Retrying...")
                time.sleep(0.1)
                continue

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
            prev_time = current_time

            # 1. Detect Face and Landmarks
            results = detector.process_frame(frame)
            landmarks = detector.extract_landmarks(results, frame.shape)
            
            if is_calibrating:
                stage = CALIBRATION_STAGES[calib_idx]
                time_left = STAGE_DURATION - (current_time - calib_start_time)
                
                # Evaluate to keep moving averages updated
                state = fatigue_system.evaluate(landmarks, frame.shape)
                
                if time_left <= 0:
                    # Move to next stage
                    calib_idx += 1
                    calib_start_time = time.time()
                    if calib_idx >= len(CALIBRATION_STAGES):
                        # Finish calibration
                        is_calibrating = False
                        
                        # Apply calibration
                        base_ear = sum(center_ear_list) / max(len(center_ear_list), 1)
                        base_mar = sum(center_mar_list) / max(len(center_mar_list), 1)
                        
                        # If list is empty due to no detection, fallback to defaults
                        if base_ear == 0: base_ear = 0.30
                        if base_mar == 0: base_mar = 0.35
                        
                        # Slightly offset the thresholds to avoid triggering during normal states. 
                        calib_ear_threshold = base_ear * 0.75 
                        calib_mar_threshold = base_mar * 1.5
                        
                        fatigue_system.set_calibration(calib_ear_threshold, calib_mar_threshold, bounds)
                    continue

                if landmarks:
                    # Only collect data in the last 2 seconds to ensure they are looking at the dot
                    if time_left < 2.0:
                        if stage == "Center":
                            center_ear_list.append(state["ear"])
                            center_mar_list.append(state["mar"])
                        else:
                            left_r, right_r, down_r = fatigue_system.get_head_ratios(landmarks)
                            if stage == "Left":
                                bounds["Left"] = max(bounds.get("Left", 0), left_r)
                            elif stage == "Right":
                                bounds["Right"] = max(bounds.get("Right", 0), right_r)
                            elif stage == "Down":
                                bounds["Down"] = max(bounds.get("Down", 0), down_r)
                                
                frame = alert_system.draw_calibration(frame, stage, max(0, time_left))
                
            else:
                # 2. Evaluate Fatigue Metrics (EAR, MAR, Head Pose, Blinks)
                state = fatigue_system.evaluate(landmarks, frame.shape)
                
                # 3. Object Detection (Phones, Bottles)
                distraction = obj_detector.scan_frame(frame)
                state["distraction"] = distraction
                
                # 4. Draw Overlays
                frame = alert_system.draw_overlays(frame, state, fps)
                
                # 5. Log Data
                session_logger.log(fps, state)

            # 6. Display Window
            cv2.imshow("Driver Fatigue Detection System", frame)

            # 5. Quit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

    obj_detector.stop()
    session_logger.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
