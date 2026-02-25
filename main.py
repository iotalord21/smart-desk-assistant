import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

start_time = time.time()
good_posture_time = 0
bad_posture_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_time = time.time()
    elapsed_time = current_time - start_time

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

        h, w, _ = frame.shape

        left_shoulder_x = int(left_shoulder.x * w)
        right_shoulder_x = int(right_shoulder.x * w)
        ear_x = int(left_ear.x * w)

        shoulder_width = abs(right_shoulder_x - left_shoulder_x)

        if shoulder_width != 0:
            forward_ratio = abs(ear_x - left_shoulder_x) / shoulder_width
        else:
            forward_ratio = 0

        if forward_ratio > 0.40:
            posture = "Bad Posture"
            color = (0, 0, 255)
            bad_posture_time += 1
        else:
            posture = "Good Posture"
            color = (0, 255, 0)
            good_posture_time += 1

        total_time = good_posture_time + bad_posture_time
        if total_time != 0:
            posture_score = (good_posture_time / total_time) * 100
        else:
            posture_score = 0

        # Display posture
        cv2.putText(frame, posture, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display ratio
        cv2.putText(frame, f"Forward Ratio: {round(forward_ratio, 2)}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display sitting time
        cv2.putText(frame, f"Session Time: {int(elapsed_time)} sec",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display posture score
        cv2.putText(frame, f"Posture Score: {int(posture_score)}%",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Break alert after 45 minutes (2700 sec)
        if elapsed_time > 2700:
            cv2.putText(frame, "TAKE A BREAK!",
                        (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Smart Desk Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()