import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

        h, w, _ = frame.shape

        shoulder_x = int(shoulder.x * w)
        ear_x = int(ear.x * w)

        forward_distance = abs(ear_x - shoulder_x)

        # Threshold (tune if needed)
        if forward_distance > 40:
            posture = "Bad Posture"
            color = (0, 0, 255)
        else:
            posture = "Good Posture"
            color = (0, 255, 0)

        cv2.putText(frame, posture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Smart Desk Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()