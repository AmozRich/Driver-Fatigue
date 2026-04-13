import numpy as np

# Config Constants (Optimized for ~15 FPS overhead from AI)
EAR_THRESHOLD = 0.22        # If lower than this, eyes are closed
MAR_THRESHOLD = 0.5         # If greater than this, yawning is detected
DROWSINESS_FRAMES = 20      # (~1.5 seconds) Prolonged closure triggers sleep warning
SLOW_BLINK_FRAMES = 8       # (~0.5s) Blinks taking longer indicates drowsiness
YAWN_FRAMES = 10            # (~0.6s) Yawn persist
HEAD_TILT_FRAMES = 20       # (~1.5s) Head tilt persist
FATIGUE_SCORE_LIMIT = 5.0   # Threshold for full driver fatigue alert

# Landmarks mapping for MediaPipe Face Mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # Right side of image
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144] # Left side of image
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178] # Used for MAR

# Iris Center (requires refine_landmarks=True)
LEFT_PUPIL = 473
RIGHT_PUPIL = 468

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_landmarks):
    # vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    # horizontal distance
    h1 = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h1)
    return ear

def calculate_mar(mouth_landmarks):
    # vertical distances
    v1 = calculate_distance(mouth_landmarks[1], mouth_landmarks[7])
    v2 = calculate_distance(mouth_landmarks[2], mouth_landmarks[6])
    v3 = calculate_distance(mouth_landmarks[3], mouth_landmarks[5])
    # horizontal distance
    h1 = calculate_distance(mouth_landmarks[0], mouth_landmarks[4])
    mar = (v1 + v2 + v3) / (3.0 * h1)
    return mar

def calculate_pupil_gaze(eye_landmarks, pupil_landmark):
    """
    Calculates where the pupil is looking horizontally.
    Returns a ratio (0.0 to 1.0) of pupil position between the inner and outer corners.
    Around 0.5 means looking straight.
    """
    # corners are the 0th and 3rd index in our eye arrays
    corner1 = eye_landmarks[0]
    corner2 = eye_landmarks[3]
    
    # Distance from corner1 to pupil
    dist_to_p = calculate_distance(corner1, pupil_landmark)
    eye_width = calculate_distance(corner1, corner2)
    
    if eye_width == 0:
        return 0.5
        
    return dist_to_p / eye_width
