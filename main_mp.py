import cv2
import numpy as np
import time
import torch
import multiprocessing
import pandas as pd
from ultralytics import YOLO
from utils_mp import update_activation_line, JuggleTracker, start_multiprocessing, track_performance
from guide import display_instruction
from data import update_google_sheet, JugglingSession
# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

#url = "https://10.41.160.144:8080/video"
# Load YOLO models with error handling
try:
    pose_model = YOLO("yolov8n-pose.engine", task="pose")
    object_model = YOLO("yolov8n.engine", task="detect")
    print("✅ YOLO models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO models: {e}")
    exit()

# Open video file
cap = cv2.VideoCapture("datasets/nub.mp4")
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open video file!")
    exit()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    data_queue, result_queue, process = start_multiprocessing()

    if not process.is_alive():
        process.start()

    BALL_CLASS_ID = 32

    # Initialize tracking variables
    ball_previous_y, ball_previous_x, ball_peak_y = None, None, None
    ball_moving_up, counting_enabled = False, False
    activation_line, JPart = None, None

    # Juggling session & tracker
    juggle_tracker = JuggleTracker()
    session = JugglingSession()

    start_time = time.time()
    guide_duration = 10
    prev_time = time.perf_counter()
    frame_count = 0
    consistency = 1.0  # Initialize consistency score

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame. Exiting loop.")
            #break

        frame_count += 1
        prev_time = track_performance(prev_time, frame_count)
        frame_count = 0  

        elapsed_time = time.time() - start_time
        if elapsed_time > guide_duration:
            counting_enabled = True  

        # Run YOLO inference with error handling
        try:
            obj_results = object_model(frame, verbose=False)
            pose_results = pose_model(frame, verbose=False)
        except Exception as e:
            print(f"❌ Error running YOLO inference: {e}")
            continue

        ball_bbox, player_bbox, keypoints = None, None, None
        ball_x, ball_y = None, None  

        # Process ball detection
        for result in obj_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, label, conf in zip(boxes, labels, confidences):
                if int(label) == BALL_CLASS_ID and conf > 0.1:
                    x1, y1, x2, y2 = map(int, box)
                    ball_bbox = np.array([x1, y1, x2, y2])
                    ball_x = (x1 + x2) // 2  
                    ball_y = y1  

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Ball ({conf:.2f})", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process pose detection
        for result in pose_results:
            keypoints = result.keypoints.xy.cpu().numpy()
            bboxes = result.boxes.xyxy.cpu().numpy()

            if len(keypoints) > 0 and len(bboxes) > 0:
                player_bbox = bboxes[0]
                if ball_bbox is not None:
                    data_queue.put((player_bbox, keypoints[0], ball_y))
                    activation_line, JPart = result_queue.get()

            frame = result.plot()

        if activation_line is not None:
            cv2.line(
                frame,
                (int(activation_line.start.x), int(activation_line.start.y)),
                (int(activation_line.end.x), int(activation_line.end.y)),
                (255, 0, 0), 2
            )

        if not counting_enabled and player_bbox is not None and activation_line is not None:
            display_instruction(frame, player_bbox, activation_line, 0)

        if counting_enabled:
            if JPart is not None:
                cv2.putText(frame, f"Tracking: {JPart}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if ball_bbox is not None:
                if ball_previous_y is not None and ball_previous_x is not None:
                    if ball_y < ball_previous_y:
                        if ball_peak_y is None or ball_y < ball_peak_y:
                            ball_peak_y = ball_y
                        ball_moving_up = True  

                    elif ball_moving_up and ball_y > ball_previous_y:
                        if ball_peak_y is not None:
                            juggle_count, time_gap = juggle_tracker.register_juggle(ball_x, ball_peak_y)
                            session.record_juggle(ball_peak_y)

                            consistency = juggle_tracker.calculate_consistency()

                            print(f"[DEBUG] Juggle: {juggle_count}, Consistency: {consistency:.2f}")

                        ball_peak_y = None  
                        ball_moving_up = False  
                
                # ✅ Get lowest ankle position (Keypoints 15 and 16)
                if keypoints is not None and len(keypoints) > 16:
                    left_ankle_y = keypoints[15][1]
                    right_ankle_y = keypoints[16][1]

                    lowest_ankle_y = max(left_ankle_y, right_ankle_y)  # Lowest point between both ankles

                    # Check if the ball is below the lowest ankle
                    if ball_y > lowest_ankle_y:
                        time_since_last_juggle = time.time() - juggle_tracker.last_juggle_time  # Time since last juggle
                        if time_since_last_juggle > 3:  # More than 3 seconds without a new juggle
                            juggle_tracker.ball_drops += 1  # Increment drop counter
                            print(f"⚠️ Ball dropped! (Ball Y: {ball_y}, Lowest Ankle Y: {lowest_ankle_y}, Time Since Last Juggle: {time_since_last_juggle:.2f}s)")

                ball_previous_y = ball_y
                ball_previous_x = ball_x  

            # Display performance metrics
            avg_time_gap = juggle_tracker.calculate_time_gap()
            total_time = juggle_tracker.get_total_time()

            cv2.putText(frame, f"Juggles: {juggle_tracker.juggle_count}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"Avg Time Gap: {avg_time_gap:.2f}s", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"Consistency: {consistency:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f"Total Time: {total_time:.2f}s", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("YOLOv8 Juggling Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            # ✅ Debug before saving
            print(f"[DEBUG] Final Values Before Saving - Time Gap: {avg_time_gap}, Consistency: {consistency}, Total Time: {total_time}")
            session.latest_consistency = consistency  # ✅ Store latest value

            # ✅ Pass juggle_tracker when saving
            session.save_to_csv(avg_time_gap=avg_time_gap, consistency=consistency, total_time=total_time)  # ✅ Now passing the required argument
            update_google_sheet()  # If using cloud storage
            break

    cap.release()
    cv2.destroyAllWindows()

    data_queue.put(None)
    if process.is_alive():
        process.join()
