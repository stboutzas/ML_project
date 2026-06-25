import numpy as np
import cv2
from pathlib import Path
from keras.models import load_model
import argparse
import sys

# def predict_single_image(model, img, target_size=(224, 224)):
#     """
#     Predicts mask/no-mask for a single image.
#     """
#     prediction = model.predict(img, verbose=1)
#     score = prediction[0][0]

#     # Determine class
#     if score < 0.5:
#         result = "MASK"
#     else:
#         result = "NO MASK"

#     print(f"   Prediction: {result}")
#     print(f"   score: {score:.4f}")

#     return {
#         'prediction': result,
#         'score': score,
#     }

def test_image(model_path, image_path, target_size=(224, 224)):
    """
    Test the model on a single image.
    """
    if not Path(image_path).exists():
        print(f"{image_path} not found")
        return None

    or_img = cv2.imread(str(image_path))
    if or_img is None:
        print(f"{image_path} error ")
        return None
    
    gray = cv2.cvtColor(or_img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Shape: (height, width, 3)

    #resized_img = cv2.resize(or_img, target_size)
    resized_img = cv2.resize(gray_3ch, target_size)
    res_displ = resized_img.copy()

    img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, target_size)
    normalized_img = resized_img.astype(np.float32) / 255.0
    # model.predict needs a batch dimension
    normalized_img = normalized_img.reshape(1, 224, 224, 3)
    #normalized_img = np.expand_dims(normalized_img, axis=0)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"{model_path} not found")
        return None

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"{model_path} error: {e}")
        return None

    #result = predict_single_image(model, normalized_img, target_size)
    prediction = model.predict(normalized_img, verbose=1)
    score = prediction[0][0]

    color = (0, 255, 0) if score < 0.5 else (0, 0, 255)  
    display_img = or_img.copy()
    cv2.putText(display_img, "MASK" if score < 0.5 else "NO MASK", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.putText(display_img, f"score: {score:.2f}", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow(f'Prediction: {Path(image_path).name}', display_img)
    cv2.imshow(f'img', res_displ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return {
        'prediction': "MASK" if score < 0.5 else "NO MASK",
        'score': score
    }

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to the image file to predict'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the trained model file (.keras)'
    )
    
    args = parser.parse_args()
    
    test_image(model_path=args.model, image_path=args.image)
    
    sys.exit(1)

if __name__ == "__main__":
    main()

# %%
