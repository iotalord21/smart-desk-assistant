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

        # Get landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]

        h, w, _ = frame.shape

        # Convert to pixel coordinates
        left_shoulder_x = int(left_shoulder.x * w)
        right_shoulder_x = int(right_shoulder.x * w)
        ear_x = int(left_ear.x * w)

        # Shoulder width (used for normalization)
        shoulder_width = abs(right_shoulder_x - left_shoulder_x)

        # Prevent division by zero
        if shoulder_width != 0:
            forward_ratio = abs(ear_x - left_shoulder_x) / shoulder_width
        else:
            forward_ratio = 0

        # Threshold for posture
        if forward_ratio > 0.40:
            posture = "Bad Posture"
            color = (0, 0, 255)
        else:
            posture = "Good Posture"
            color = (0, 255, 0)

        # Display posture result
        cv2.putText(frame, posture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display forward ratio value
        cv2.putText(frame, f"Forward Ratio: {round(forward_ratio, 2)}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Smart Desk Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()