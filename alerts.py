import cv2
import winsound
import os

class AlertSystem:
    def __init__(self):
        self.alarm_playing = False
        # Path for custom audio file. If it doesn't exist, it will fallback to system beep.
        self.audio_file = "alarm.wav"

    def trigger_audio(self, play):
        if play and not self.alarm_playing:
            # SND_ASYNC means it returns immediately so video doesn't lag
            # SND_LOOP means it repeats until stopped
            # SND_NODEFAULT means if file is missing, it fails silently, but here we want fallback so no NODEFAULT
            flags = winsound.SND_ASYNC | winsound.SND_LOOP
            if os.path.exists(self.audio_file):
                winsound.PlaySound(self.audio_file, winsound.SND_FILENAME | flags)
            else:
                # Fallback to a system sound if alarm.wav isn't present
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | flags)
            self.alarm_playing = True
        elif not play and self.alarm_playing:
            # Stop the sound
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.alarm_playing = False

    def draw_calibration(self, frame, stage, time_left):
        """Draws the calibration UI."""
        h, w = frame.shape[:2]
        
        # Dim the background a bit for focus
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "CALIBRATION MODE", (w//2 - 150, 50), font, 1.0, (255, 255, 0), 2)
        cv2.putText(frame, "Turn your whole head to look at the red dot.", (w//2 - 250, 90), font, 0.7, (255, 255, 255), 2)
        
        # Determine dot position
        dot_pos = (w//2, h//2)
        if stage == "Left":
            dot_pos = (50, h//2)
        elif stage == "Right":
            dot_pos = (w - 50, h//2)
        elif stage == "Down":
            dot_pos = (w//2, h - 50)
            
        # Draw the dot
        cv2.circle(frame, dot_pos, 20, (0, 0, 255), -1)
        cv2.circle(frame, dot_pos, 25, (0, 255, 255), 2) # Outer ring
        
        # Time left
        cv2.putText(frame, f"Hold for {int(time_left)}s", (w//2 - 80, h - 30), font, 0.8, (0, 255, 0), 2)
        return frame

    def draw_overlays(self, frame, state, fps):
        h, w = frame.shape[:2]
        
        # Draw a transparent dark background for the dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 180), (0, 0, 0), -1)
        # Apply the transparent overlay
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        
        if not state["driver_detected"]:
            # Flash driver not detected
            cv2.putText(frame, "DRIVER NOT DETECTED", (w//2 - 180, h//2), font, 1.0, (0, 0, 255), 3)
            return frame

        # Assign color based on the status severity
        status_color = (0, 255, 0) # Green 
        if state["status"] == "Warning":
            status_color = (0, 255, 255) # Yellow
        elif state["status"] == "DROWSINESS ALERT":
            status_color = (0, 0, 255) # Red

        # Render stats dashboard
        cv2.putText(frame, f"Blinks/Min: {state['blinks_per_min']}", (20, 40), font, scale, (255, 255, 255), thickness)
        cv2.putText(frame, f"Head: {state['head_pose']}", (20, 70), font, scale, (255, 255, 255), thickness)
        
        # Color the gaze differently if distracted
        gaze_color = (255, 255, 255)
        if state.get("pupil_gaze", "Center") in ["Left Distracted", "Right Distracted"]:
            gaze_color = (0, 165, 255) # Orange
            
        cv2.putText(frame, f"Gaze: {state.get('pupil_gaze', 'Center')}", (20, 100), font, scale, gaze_color, thickness)
        cv2.putText(frame, f"Status: {state['status']}", (20, 130), font, scale, status_color, thickness)
        cv2.putText(frame, f"Fatigue Score: {state['fatigue_score']}", (20, 160), font, scale, (255, 255, 255), thickness)
        
        # Show physical distraction warnings (YOLOv8)
        distraction = state.get("distraction", "None")
        if distraction != "None":
            cv2.rectangle(frame, (w - 300, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"DISTRACTION: {distraction}", (w - 280, h - 35), font, 0.7, (0, 255, 255), 2)
            # Maybe also beep lightly
            if not self.alarm_playing:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
        
        # Show central strong warning if the score triggers an alert
        if state["status"] == "DROWSINESS ALERT":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4) # Red border around screen
            cv2.putText(frame, "Wake up! Drowsiness Detected", (w//2 - 200, h//2), font, 0.9, (0, 0, 255), 3)
            self.trigger_audio(True)
        else:
            self.trigger_audio(False)
        
        # Show FPS at the top right
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 40), font, scale, (255, 255, 0), thickness)
        
        return frame
