import cv2
import numpy as np
import time
from collections import deque
from utils import *

class FatigueDetector:
    def __init__(self):
        self.closed_eyes_frames = 0
        self.yawn_frames = 0
        self.head_tilt_frames = 0
        self.blink_active_frames = 0
        self.distracted_gaze_frames = 0
        
        self.fatigue_score = 0.0
        self.last_natural_decay_time = time.time()
        self.last_hyper_blink_time = time.time()
        self.last_blink_time = time.time() # For blink cooldown
        
        # Event cooldowns (prevent double jeopardy on jitter)
        self.event_cooldowns = {
            "yawning": 0.0,
            "head_tilt": 0.0,
            "slow_blink": 0.0,
            "distracted_gaze": 0.0
        }
        
        # Dynamic thresholds (will be overwritten if calibration runs)
        self.ear_threshold = EAR_THRESHOLD
        self.mar_threshold = MAR_THRESHOLD
        
        # Default head pose ratios (will be calibrated to the specific user's range of motion)
        self.head_left_ratio = 1.5
        self.head_right_ratio = 1.5
        self.head_down_ratio = 1.5
        
        # Signal Smoothing (Simple Moving Average over 5 frames)
        self.ear_history = deque(maxlen=5)
        self.mar_history = deque(maxlen=5)
        
        self.state = {
            "driver_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "head_pose": "Normal",
            "fatigue_score": 0.0,
            "status": "Alert",
            "blinks_per_min": 0
        }
        self.blink_timestamps = []
        
        # Event flags ensure we only punish an event ONCE per continuous action
        self.event_flags = {
            "yawning": False,
            "head_tilt": False,
            "drowsy_eyes": False,
            "slow_blink": False,
            "distracted_gaze": False
        }

    def evaluate(self, landmarks, frame_shape):
        current_time = time.time()
        
        # 1. Natural Score Decay (Score slowly goes down if driver pays attention)
        if self.fatigue_score > 0 and (current_time - self.last_natural_decay_time) > 5.0:
            self.fatigue_score -= 0.5
            if self.fatigue_score < 0: self.fatigue_score = 0.0
            self.last_natural_decay_time = current_time

        if not landmarks:
            self.state["driver_detected"] = False
            self.state["status"] = "Driver Not Detected"
            return self.state

        self.state["driver_detected"] = True

        # ========================================
        #  Eye Tracking (EAR & Blinks)
        # ========================================
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
        raw_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        
        # Signal Smoothing
        self.ear_history.append(raw_ear)
        smoothed_ear = sum(self.ear_history) / len(self.ear_history)
        self.state["ear"] = round(smoothed_ear, 2)

        # Head Pose Guard: ONLY track blinks if the user is looking relatively forward
        # This prevents rapid false blinks when the face is turned away
        head_pose = self.estimate_head_pose_2d(landmarks) 
        self.state["head_pose"] = head_pose
        
        if head_pose == "Normal":
            if smoothed_ear < self.ear_threshold:
                self.closed_eyes_frames += 1
                self.blink_active_frames += 1
            else:
                # Eyes just opened! Meaning the blink just finished.
                # Enforce a 0.2 second blink cooldown to prevent biological impossibilities / noise
                if self.blink_active_frames > 0 and (current_time - self.last_blink_time) > 0.2:
                    self.blink_timestamps.append(current_time)
                    self.last_blink_time = current_time
                    # Slow blink penalty
                    if self.blink_active_frames >= SLOW_BLINK_FRAMES and not self.event_flags["slow_blink"]:
                        if current_time - self.event_cooldowns["slow_blink"] > 3.0:
                            self.fatigue_score += 1.0
                            self.event_cooldowns["slow_blink"] = current_time
                        self.event_flags["slow_blink"] = True
                
                # Reset active eye tracking states
                self.closed_eyes_frames = 0
                self.blink_active_frames = 0
                self.event_flags["drowsy_eyes"] = False
                self.event_flags["slow_blink"] = False
                
            # ========================================
            #  Pupil / Iris Tracking (Side-Eye)
            # ========================================
            # Only reliably track iris if the user's eyes aren't closed
            if smoothed_ear > self.ear_threshold:
                # Need to safely check if we have 478 landmarks (refine_landmarks=True)
                if len(landmarks) > RIGHT_PUPIL and len(landmarks) > LEFT_PUPIL:
                    right_pupil = landmarks[RIGHT_PUPIL]
                    left_pupil = landmarks[LEFT_PUPIL]
                    
                    # Ratios > 0.70 or < 0.30 generally mean side-eye
                    r_gaze = calculate_pupil_gaze(right_eye, right_pupil)
                    l_gaze = calculate_pupil_gaze(left_eye, left_pupil)
                    avg_gaze = (r_gaze + l_gaze) / 2.0
                    
                    if avg_gaze < 0.30:
                        self.state["pupil_gaze"] = "Left Distracted"
                    elif avg_gaze > 0.70:
                        self.state["pupil_gaze"] = "Right Distracted"
                    else:
                        self.state["pupil_gaze"] = "Center"
                        
                    # Penalize distracted gaze
                    if self.state["pupil_gaze"] != "Center":
                        self.distracted_gaze_frames += 1
                        # If looking away for ~1.5 seconds continuously at 15FPS
                        if self.distracted_gaze_frames >= 20:
                            if not self.event_flags["distracted_gaze"]:
                                if current_time - self.event_cooldowns["distracted_gaze"] > 3.0:
                                    self.fatigue_score += 1.0  # Add penalty once
                                    self.event_cooldowns["distracted_gaze"] = current_time
                                self.event_flags["distracted_gaze"] = True
                    else:
                        self.distracted_gaze_frames = 0
                        self.event_flags["distracted_gaze"] = False
            else:
                self.state["pupil_gaze"] = "Closed"
                self.distracted_gaze_frames = 0
                self.event_flags["distracted_gaze"] = False
        else:
            # If head is turned, reset blink frames so it doesn't accidentally trigger drowsiness
            self.closed_eyes_frames = 0
            self.blink_active_frames = 0
            self.state["pupil_gaze"] = "Not Visible"

        # Prolonged Eye Closure (+3)
        if self.closed_eyes_frames >= DROWSINESS_FRAMES:
            if not self.event_flags["drowsy_eyes"]:
                self.fatigue_score += 3.0
                self.event_flags["drowsy_eyes"] = True

        # Blinks Per Min Tracking
        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 60.0]
        self.state["blinks_per_min"] = len(self.blink_timestamps)
        
        # Flag hyper-frequent blinking (stress/distraction)
        if len(self.blink_timestamps) > 40:
            if current_time - self.last_hyper_blink_time > 2.0:
                self.fatigue_score += 0.2
                self.last_hyper_blink_time = current_time

        # ========================================
        #  Mouth Tracking (Yawning)
        # ========================================
        mouth = [landmarks[i] for i in MOUTH_INDICES]
        raw_mar = calculate_mar(mouth)
        
        # Signal Smoothing
        self.mar_history.append(raw_mar)
        smoothed_mar = sum(self.mar_history) / len(self.mar_history)
        self.state["mar"] = round(smoothed_mar, 2)

        if smoothed_mar > self.mar_threshold:
            self.yawn_frames += 1
            if self.yawn_frames >= YAWN_FRAMES:
                if not self.event_flags["yawning"]:
                    if current_time - self.event_cooldowns["yawning"] > 3.0:
                        self.fatigue_score += 2.0
                        self.event_cooldowns["yawning"] = current_time
                    self.event_flags["yawning"] = True
        else:
            self.yawn_frames = 0
            self.event_flags["yawning"] = False

        # ========================================
        #  Head Pose Tracking
        # ========================================
        # Head pose was already calculated above for the blink guard!

        if head_pose != "Normal":
            self.head_tilt_frames += 1
            if self.head_tilt_frames >= HEAD_TILT_FRAMES:
                if not self.event_flags["head_tilt"]:
                    if current_time - self.event_cooldowns["head_tilt"] > 3.0:
                        self.fatigue_score += 2.0
                        self.event_cooldowns["head_tilt"] = current_time
                    self.event_flags["head_tilt"] = True
        else:
            self.head_tilt_frames = 0
            self.event_flags["head_tilt"] = False


        # ========================================
        #  Determine Status
        # ========================================
        self.state["fatigue_score"] = round(self.fatigue_score, 1)

        # Trigger conditions
        if self.fatigue_score >= FATIGUE_SCORE_LIMIT or self.closed_eyes_frames >= DROWSINESS_FRAMES:
            self.state["status"] = "DROWSINESS ALERT"
        elif self.fatigue_score >= FATIGUE_SCORE_LIMIT / 2:
            self.state["status"] = "Warning"
        else:
            self.state["status"] = "Alert"

        return self.state


    def estimate_head_pose_2d(self, landmarks):
        """Uses facial proportion heuristics for rock-solid stability"""
        nose = landmarks[1]
        left_side = landmarks[234]  # Extent of left cheek
        right_side = landmarks[454] # Extent of right cheek
        chin = landmarks[152]
        forehead = landmarks[10]
        
        # Calculate proportional distances
        left_dist = calculate_distance(nose, left_side)
        right_dist = calculate_distance(nose, right_side)
        top_dist = calculate_distance(nose, forehead)
        bottom_dist = calculate_distance(nose, chin)

        # Evaluate Yaw (Turned Left or Right)
        # We process this against the mirrored camera behavior to match physical motion
        if left_dist > right_dist * self.head_left_ratio:
            return "Looking Left"    # Flipped back to match physical left 
        elif right_dist > left_dist * self.head_right_ratio:
            return "Looking Right"   # Flipped back to match physical right

        # Evaluate Pitch (Looking Down)
        if top_dist > bottom_dist * self.head_down_ratio:
            return "Looking Down"

        return "Normal"

    def set_calibration(self, new_ear, new_mar, bounds):
        """Sets customized, driver-specific thresholds."""
        # Fix the Talking=Yawning bug using a safe ceiling/floor.
        protected_ear = min(max(new_ear, 0.15), 0.25)
        protected_mar = max(new_mar, 0.35) # Lowered floor from 0.45 so average mouths can still trigger Yawn
        
        print(f"Calibration applied: EAR={protected_ear}, MAR={protected_mar}, Bounds={bounds}")
        self.ear_threshold = protected_ear
        self.mar_threshold = protected_mar
        
        # Override the thresholds with a slight margin
        if "Left" in bounds:
            self.head_left_ratio = bounds["Left"] * 0.8  # trigger slightly before they reach max
        if "Right" in bounds:
            self.head_right_ratio = bounds["Right"] * 0.8
        if "Down" in bounds:
            self.head_down_ratio = bounds["Down"] * 0.8

    def get_head_ratios(self, landmarks):
        """Returns the proportion of left/right and top/bottom distances to determine bounds."""
        if not landmarks: return 1.0, 1.0, 1.0
        nose = landmarks[1]
        left_side = landmarks[234]
        right_side = landmarks[454]
        chin = landmarks[152]
        forehead = landmarks[10]
        
        left_dist = calculate_distance(nose, left_side)
        right_dist = calculate_distance(nose, right_side)
        top_dist = calculate_distance(nose, forehead)
        bottom_dist = calculate_distance(nose, chin)
        
        right_d = max(right_dist, 1)
        left_d = max(left_dist, 1)
        bottom_d = max(bottom_dist, 1)
        
        return left_dist / right_d, right_dist / left_d, top_dist / bottom_d
