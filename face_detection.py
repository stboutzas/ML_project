import cv2
import sys
from pathlib import Path

def detect_and_draw_faces(image_path, show_image=False):
    """
    Detect faces in an image and draw green rectangles around them.
    
    Args:
        image_path: Path to the image file
        show_image: Whether to display the image with detected faces

    """
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    if show_image:
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Print results
        print(f"📸 Image: {Path(image_path).name}")
        print(f"👤 Faces detected: {len(faces)}")
        
        # Display image
        cv2.imshow(f'Face Detection: {Path(image_path).name}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return len(faces)

# ============================================
# COMMAND LINE USAGE
# ============================================

if __name__ == "__main__":

    # Check if image path was provided
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <image_path>")
    #     print("Example: python script.py face.jpg")
    #     sys.exit(1)
        
    # Iterate through all files in all subfolders
    i = 0
    for file_path in Path('files/FaceMaskDataset').rglob('*.*'):
        if file_path.is_file():
            #print(file_path)
            detected_faces = detect_and_draw_faces(file_path)
            if detected_faces is not None and detected_faces < 1:
                print(f"❌ No faces detected in {file_path}. ")
                i += 1
            elif detected_faces is None:
                print(f"❌ Could not process {file_path.name}. ")
    
    print(i)
    # image_path = sys.argv[1]
    # detected_faces = detect_and_draw_faces(image_path)
    # if detected_faces < 1:
    #     print("❌ No faces detected in {file_path.name}. ")