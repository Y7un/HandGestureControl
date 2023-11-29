import time
import cv2
import mediapipe as mp
import pyautogui

# Constants
GESTURE_TIME_THRESHOLD = 0.2
RIGHT_KEY = "right"
LEFT_KEY = "left"
UP_KEY = "up"
DOWN_KEY = "down"
SPACE_KEY = "space"

def detect_gesture(hand_landmarks):
    cnt = 0
    thresh = (hand_landmarks.landmark[0].y * 100 - hand_landmarks.landmark[9].y * 100) / 2

    if (hand_landmarks.landmark[5].y * 100 - hand_landmarks.landmark[8].y * 100) > thresh:
        cnt += 1

    # Repeat similar checks for other fingers

    if (hand_landmarks.landmark[5].x * 100 - hand_landmarks.landmark[4].x * 100) > 5:
        cnt += 1

    return cnt

cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1

while True:
    end_time = time.time()
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    res = hand_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        hand_keypoints = res.multi_hand_landmarks[0]
        cnt = detect_gesture(hand_keypoints)

        if not (prev == cnt):
            if not start_init:
                start_time = time.time()
                start_init = True

            elif (end_time - start_time) > GESTURE_TIME_THRESHOLD:
                if cnt == 1:
                    pyautogui.press(RIGHT_KEY)
                elif cnt == 2:
                    pyautogui.press(LEFT_KEY)
                elif cnt == 3:
                    pyautogui.press(UP_KEY)
                elif cnt == 4:
                    pyautogui.press(DOWN_KEY)
                elif cnt == 5:
                    pyautogui.press(SPACE_KEY)

                prev = cnt
                start_init = False

        drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], hands.HAND_CONNECTIONS)

    cv2.imshow("my_window", frame)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break

