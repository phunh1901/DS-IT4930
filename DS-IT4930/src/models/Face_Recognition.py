import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import matplotlib.pyplot as plt
import pickle
import os


class FaceVerification:
    # init
    def __init__(self):
        self.detector = MTCNN()
        self.facenet = FaceNet()
       
    # detect and extract    
    def detect_and_extract_face(self, image):
        results = self.detector.detect_faces(image)
        
        if len(results) == 0:
            return None
        
        face = max(results, key=lambda x: x['confidence'])
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        
        return face_img
    
    # embedding img
    def get_embedding(self, face_image):
        if face_image is None:
            return None
            
        face_image = face_image.astype('float32')
        face_image = np.expand_dims(face_image, axis=0)
        embedding = self.facenet.embeddings(face_image)[0]
        
        return embedding
    
    # calculate distance
    def calculate_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    
    # verify (dùng cho so sánh 2 ảnh)
    def verify(self, image1, image2):
        face1 = self.detect_and_extract_face(image1)
        face2 = self.detect_and_extract_face(image2)
        
        if face1 is None or face2 is None:
            return None, None, face1, face2
        
        embedding1 = self.get_embedding(face1)
        embedding2 = self.get_embedding(face2)
        
        distance = self.calculate_distance(embedding1, embedding2)
        
        return distance, face1, face2

# Register    
def Face_Register(images, infor, data_file="data.pkl"):
    embeddings = []
    register = FaceVerification()
    
    for i, img in enumerate(images):
        if isinstance(img, str):
            img = cv2.imread(img)
        
        # Chuyển BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face = register.detect_and_extract_face(img_rgb)
        
        # KIỂM TRA face trước khi dùng
        if face is None:
            print(f"Warning: No face detected in image {i+1}, skipping...")
            continue
        
        emb = register.get_embedding(face)
        
        if emb is not None:
            embeddings.append(emb)
    
    # Kiểm tra có ít nhất 1 embedding
    if len(embeddings) == 0:
        print("Error: No faces detected in any images!")
        return None
    
    final_embedding = np.mean(embeddings, axis=0)
    
    if os.path.exists(data_file):
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print(" Warning: data.pkl is corrupted, creating new file...")
            data = {}
    else:
        data = {}
        
    student_id = infor["student_id"]
    data[student_id] = {
        "name": infor["name"],
        "embedding": final_embedding.tolist()
    }
    
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    
    print("Register successful!")
    print(f"Used {len(embeddings)}/{len(images)} images")
    return final_embedding

# Verification from cam        
def Face_Verification(image, data_file="data.pkl", threshold=0.75):
    if image is None:
        print("Read image Error!")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    verifier = FaceVerification()
    
    #SỬA: Phải detect face trước
    face_input = verifier.detect_and_extract_face(image_rgb)
    
    if face_input is None:
        print("No face detected in frame")
        return None
    
    # Lấy embedding từ face đã detect
    emb_input = verifier.get_embedding(face_input)
    
    if emb_input is None:
        print("Cannot get embedding")
        return None
    
    try:
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    except:
        print("Load data Error")
        return None
    
    best_id = None
    best_name = None
    best_distance = 999
    
    for student_id, infor in data.items():
        emb_db = np.array(infor['embedding'], dtype=float)
        
        distance = verifier.calculate_distance(emb_input, emb_db)
        
        if distance < best_distance:
            best_distance = distance
            best_id = student_id
            best_name = infor.get("name", "Unknown")
    
    is_same = best_distance < threshold
    
    return {
        "student_id": best_id,
        "name": best_name,
        "distance": best_distance,
        "match": is_same
    }

def main():
    print("-" *60)
    
    
    while True:
        choice = int(input("Register (1), Verify (2), Out (0) Which choice?: "))
        if choice == 1:
            print("📝 Registration Mode")
            print("Please provide 3 images:\n")
    
            raw_path1 = input("Path1: ").strip()
            raw_path2 = input("Path2: ").strip()
            raw_path3 = input("Path3: ").strip()
    
            if not raw_path1 or not raw_path2 or not raw_path3:
                print("Path Error: One or more paths are empty")
                return None
    
            image1 = cv2.imread(raw_path1, cv2.IMREAD_COLOR)
            image2 = cv2.imread(raw_path2, cv2.IMREAD_COLOR)
            image3 = cv2.imread(raw_path3, cv2.IMREAD_COLOR)
    
            if image1 is None or image2 is None or image3 is None:
                print("Read image Error: Cannot read one or more images")
                return None
            
            images = [image1, image2, image3]
        
            student_id_input = input("Student ID: ").strip()
            name_input = input("Name: ").strip()
    
            info = {
                'student_id': student_id_input,
                'name': name_input
            }

            Face_Register(images, info)
        
        elif choice == 2:
            print("Verification Mode")
            print("Press SPACE to capture and verify")
            print("Press ESC to exit\n")
        
            cap = cv2.VideoCapture(0)
    
            if not cap.isOpened():
                print("Cam Error")
                return
    
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Cam Error")
                    break
        
                cv2.imshow('Camera - Press SPACE to verify', frame)
        
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                if key == 32:  # SPACE
                    print("Verifying...")
                    result = Face_Verification(frame)

                    if result is None:
                        print("Verification failed")
                        continue
            
                    print("\n----- Result -----")
                    print(f"Full Name: {result['name']}")
                    print(f"ID: {result['student_id']}")
                    print(f"Distance: {result['distance']:.4f}")
                    print(f"Conclusion: {'Match!' if result['match'] else 'No Match'}")
                    print("---------------------\n")
                
            cap.release()
            cv2.destroyAllWindows()
            
        else:
            return   
        print("-" *60)

if __name__ == "__main__":
    main()
    
    
