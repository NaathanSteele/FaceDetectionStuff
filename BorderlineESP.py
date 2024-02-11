import cv2
import mediapipe as mp
import time
from mtcnn import MTCNN
import asyncio

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.start_time = time.time()
        self.frame_width = None
        self.frame_height = None
    
    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks
    
    def draw_landmarks(self, frame, hand_landmarks):
        if self.frame_width is None or self.frame_height is None:
            self.frame_height, self.frame_width, _ = frame.shape
        
        for hand_landmark in hand_landmarks:
            x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0
            for landmark in hand_landmark.landmark:
                x = int(landmark.x * self.frame_width)
                y = int(landmark.y * self.frame_height)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 0, 255), 2)

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        return faces
    
    def draw_faces(self, frame, faces):
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

async def main():
    video_capture = cv2.VideoCapture(0)
    hand_tracker = HandTracker()
    face_detector = FaceDetector()
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize the frame to a smaller resolution
        frame = cv2.resize(frame, (640, 480))
        
        # Perform hand tracking and face detection concurrently
        hand_landmarks = hand_tracker.detect_hands(frame)
        faces = face_detector.detect_faces(frame)
        
        # Draw landmarks and faces on the frame
        if hand_landmarks:
            hand_tracker.draw_landmarks(frame, hand_landmarks)  # Draw landmarks for all detected hands
        face_detector.draw_faces(frame, faces)
        
        current_time = time.time()
        elapsed_time = current_time - hand_tracker.start_time
        
        cv2.putText(frame, f"App Running: {elapsed_time:.2f}s", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "v.1", (frame.shape[1] - 100, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Version 1 - Uncooked Spagetti', frame)
        
        fps_frame_count += 1
        if current_time - fps_start_time >= 1:
            fps = fps_frame_count / (current_time - fps_start_time)
            print(f"FPS: {fps:.2f}")
            fps_frame_count = 0
            fps_start_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
