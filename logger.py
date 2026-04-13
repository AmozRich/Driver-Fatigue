import csv
import time
from datetime import datetime
import os

class SessionLogger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"session_{timestamp}.csv")
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write header
        self.writer.writerow([
            "Timestamp", 
            "FPS", 
            "EAR", 
            "MAR", 
            "Head_Pose", 
            "Pupil_Gaze",
            "Blinks_Per_Min", 
            "Fatigue_Score", 
            "Status",
            "Distraction_Object"
        ])
        
    def log(self, fps, state):
        if not self.file or self.file.closed:
            return
            
        current_time = datetime.now().strftime("%H:%M:%S")
        self.writer.writerow([
            current_time,
            round(fps, 1),
            state.get("ear", 0.0),
            state.get("mar", 0.0),
            state.get("head_pose", "Normal"),
            state.get("pupil_gaze", "Center"),
            state.get("blinks_per_min", 0),
            state.get("fatigue_score", 0.0),
            state.get("status", "Alert"),
            state.get("distraction", "None")
        ])
        self.file.flush()
        
    def close(self):
        if self.file and not self.file.closed:
            self.file.close()

