import cv2
import mediapipe as mp
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

finger_state = False

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                screen_width, screen_height = pyautogui.size()
                x = int(x * screen_width)
                y = int(y * screen_height)

                pyautogui.moveTo(x, y)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if calculate_distance(thumb_tip, index_tip) < 0.05:
                    if not finger_state:
                        pyautogui.mouseDown()
                        finger_state = True
                else:
                    if finger_state:
                        pyautogui.mouseUp()
                        finger_state = False

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
