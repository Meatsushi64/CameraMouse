import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture camera input
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
# width: 2880
# height: 1620

def width_change(x):
    return x * 2/3 * screen_width + 1/6 * screen_width
def height_change(y):
    return y * 2/3 * screen_height + 1/6 * screen_height


buffer_size = 5
positions = []


while True:
    # Step 3: Capture camera frame
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 4: Process hand landmarks
    results = hands.process(img_rgb)
    
    # Step 5: Map hand position to screen coordinates
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position
            finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]


            # Map finger position to screen coordinates
            x, y = int(middle_mcp.x * screen_width), int(middle_mcp.y * screen_height)
            # Add the new position to the buffer
            positions.append((x, y))

            # Keep the buffer size constant
            if len(positions) > buffer_size:
                positions.pop(0)  # Remove the oldest position

            # Step 8: Smooth movement by averaging positions in the buffer
            avg_x = int(np.mean([pos[0] for pos in positions]))
            avg_y = int(np.mean([pos[1] for pos in positions]))
            
            print(str(avg_x) + " and " + str(avg_y))


            # Step 6: Move the mouse based on hand position
            # print(str() + " and " + str(screen_width))
            pyautogui.moveTo(avg_x, avg_y)

            # Step 7: Check distance between thumb and finger tip. if close, then click
            distance = ((thumb_tip.x - finger_tip.x)**2 + (thumb_tip.y - finger_tip.y)**2)**0.5
            if distance < 0.05:  # Detects if fingers are close together
                pyautogui.click()
    
    # Display the camera feed
    cv2.imshow("Hand Tracking", img)
    
    # Exit the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
