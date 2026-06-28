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

def process_frame(frame, target_size=(224, 224), grayScale=False):

    # img = cv2.imread(str(image_path))
    img = frame.copy()
    # ////////// ===>>>>>>> not converted to RGB !!!!!!!!!!!!
    if grayScale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Shape: (height, width, 3)
    # ////////// !!!!!!!!!!!!

    resized_img = cv2.resize(img, target_size)
    normalized_img = resized_img.astype(np.float32) / 255.0
    converted_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    batch = np.reshape(converted_img, (1, target_size[0], target_size[1], 3))
    return batch, resized_img # return also resized image to show for debug

TIME_BUFFER = 0#0.3  # seconds
    
def main():
  
    # args
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the trained model file (.keras)'
    )
    
    args = parser.parse_args()
    
    # load model
    model = load_model_from_path(args.model)
    if model is None:
        return None
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera is now set to {actual_width}x{actual_height}")

    last_capture_time = -50
    #captured_frame = None

    # ret, frame = cap.read()
    # frame = cv2.flip(frame, 1, 1)

    # captured_frame = frame.copy()
    i = 1

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    
    #return len(faces)

    while(True):
        ret, frame = cap.read()
        
        if cv2.waitKey(33) & 0xFF == ord('\x1b'):
            break

        # press c to capture frame for debugging
        if cv2.waitKey(33) & 0xFF == ord('c'):
            filename = f"captured_{i}.jpg"
            i += 1
            cv2.imwrite(filename, frame)
            print(f"captured: {filename}")

        current_time = time.time()
        if current_time - last_capture_time >= TIME_BUFFER:

            captured_frame = frame.copy()
            last_capture_time = current_time

            #faces = face_cascade.detectMultiScale(captured_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = face_cascade.detectMultiScale(captured_frame)

            face_img = captured_frame.copy()  # Default to the whole frame if no faces are detected
            if(len(faces) == 1):
                (x, y, w, h) = faces[0]
                face_img = captured_frame[y:y+h, x:x+w].copy()  # Crop the face region with some padding
                cv2.rectangle(captured_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # for (x, y, w, h) in faces:
            #     face_img = captured_frame[y-20:y+h+20, x-20:x+w+20].copy()  # Crop the face region with some padding
            #     cv2.rectangle(captured_frame, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)

            processed_frame, resized_img = process_frame(face_img)#, grayScale=True)  # Change to True if your model expects grayscale input
            prediction = model.predict(processed_frame, verbose=0)
            score = prediction[0][0]

            text = 'MASK' if score < 0.5 else 'NO MASK'
            color = (0, 255, 0) if score < 0.5 else (0, 0, 255)


        # if captured_frame is None:
        #     captured_frame = np.zeros_like(frame)  # Black frame

            cv2.putText(captured_frame, str(len(faces)), (620, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv2.putText(captured_frame, text, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            cv2.putText(captured_frame, str(score), (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            
            #display = np.hstack((frame, pred['img']))
            # display_img = (pred['img'][0] * 255).astype(np.uint8)
            # display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

            #cv2.imshow('frame', frame)
            cv2.imshow('captured_frame', captured_frame)
            cv2.imshow('resized_frame', resized_img)  # Show the resized frame

    cap.release()
    cv2.destroyAllWindows()

    sys.exit(1)

if __name__ == "__main__":
    main()