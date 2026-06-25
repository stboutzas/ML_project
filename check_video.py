import sys

import cv2 
from matplotlib.pyplot import gray
import numpy as np
import time
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

def load_model_from_path(model_path):
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"{model_path} not found")
        return None
    
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"{model_path} error: {e}")
        return None
    return model

def process_frame(frame, target_size=(224, 224)):

    # img = cv2.imread(str(image_path))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Shape: (height, width, 3)

    #resized_img = cv2.resize(frame, target_size)
    resized_img = cv2.resize(gray_3ch, target_size)
    # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # img = rgb_img.astype(np.float32) / 255.0
    img = resized_img.astype(np.float32) / 255.0
    batch = np.reshape(img, (1, target_size[0], target_size[1], 3))
    return batch, resized_img

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the trained model file (.keras)'
    )
    
    args = parser.parse_args()
    
    model = load_model_from_path(args.model)
    if model is None:
        return None
    
    cap = cv2.VideoCapture(0)
    # After setting, check the actual width and height
    # Set width to 640 pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # Set height to 480 pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera is now set to {actual_width}x{actual_height}")

    last_capture_time = -50
    #captured_frame = None

    ret, frame = cap.read()
    captured_frame = frame.copy()
    i = 1
    while(True):
        ret, frame = cap.read()
        
        if cv2.waitKey(33) & 0xFF == ord('\x1b'):
            break

        if cv2.waitKey(33) & 0xFF == ord('c'):
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # filename = f"captured_{timestamp}.jpg"
            filename = f"captured_{i}.jpg"
            i += 1
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',gray)
        #cv2.imshow('frame',frame)

        current_time = time.time()
        if current_time - last_capture_time >= 0.3:
            captured_frame = frame.copy()
            last_capture_time = current_time

            
            processed_frame, resized_img = process_frame(captured_frame)
            prediction = model.predict(processed_frame, verbose=0)
            score = prediction[0][0]

            text = 'MASK' if score < 0.5 else 'NO MASK'
            color = (0, 255, 0) if score < 0.5 else (0, 0, 255)

        # Stack images side by side
        if captured_frame is None:
            captured_frame = np.zeros_like(frame)  # Black frame

       
        cv2.putText(captured_frame, text, (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.putText(captured_frame, str(score), (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        #display = np.hstack((frame, pred['img']))
        # display_img = (pred['img'][0] * 255).astype(np.uint8)
        # display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)
        cv2.imshow('captured_frame', captured_frame)
        cv2.imshow('resized_frame', resized_img)  # Show the resized frame

    cap.release()
    cv2.destroyAllWindows()

    sys.exit(1)

if __name__ == "__main__":
    main()