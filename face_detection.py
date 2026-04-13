import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Gets better precision around eyes
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results

    def extract_landmarks(self, results, frame_shape):
        h, w, _ = frame_shape
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        points = []
        for lm in face_landmarks.landmark:
            # Scale landmarks to image dimensions
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
        return points
