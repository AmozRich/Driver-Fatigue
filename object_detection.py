import cv2
import threading
import time
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        # Loads the smallest, fastest model
        print("Loading YOLOv8 model for Distraction Detection...")
        self.model = YOLO("yolov8n.pt")
        
        # COCO Dataset Classes: 39=bottle, 41=cup, 67=cell phone
        self.target_classes = [39, 41, 67]
        self.last_detection = "None"
        
        # Threading mechanisms
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Start the background AI worker
        self.thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.thread.start()
        
    def scan_frame(self, frame):
        """
        Called 30-60 times a second by the main loop. 
        Only copies the frame when the background thread is ready for a new one.
        Returns the most recently completed YOLO result.
        """
        with self.lock:
            # Only copy if the thread has consumed the previous frame
            if self.latest_frame is None:
                self.latest_frame = frame.copy()
            current_detection = self.last_detection
            
        return current_detection
            
    def _scan_loop(self):
        """
        The background thread loop. It runs exclusively on a separate CPU core/GPU thread.
        It pulls the most recent frame, runs the heavy YOLO math, and posts the result.
        """
        while self.running:
            frame_to_process = None
            
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None # Consume the frame so we don't scan it twice in a row
                    
            if frame_to_process is not None:
                # Run prediction on the frame (Conf=0.45 threshold)
                results = self.model(frame_to_process, classes=self.target_classes, conf=0.45, verbose=False)
                new_detection = "None"
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        new_detection = results[0].names[cls_id].upper()
                        break
                        
                with self.lock:
                    self.last_detection = new_detection
            else:
                # If no frame is ready, sleep for 10ms to prevent 100% CPU starvation
                time.sleep(0.01)
                
    def stop(self):
        """Gracefully shuts down the background worker."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
