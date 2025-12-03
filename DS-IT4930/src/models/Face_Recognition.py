import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import matplotlib.pyplot as plt
import pickle
import os
import shutil

# Định nghĩa đường dẫn
DATA_DIR = "../../data/Face_Verification"
DATA_FILE = os.path.join(DATA_DIR, "data.pkl")
PERSONAL_DIR = os.path.join(DATA_DIR, "Personal")

# Tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSONAL_DIR, exist_ok=True)

print(f"Data directory: {os.path.abspath(DATA_DIR)}")
print(f"Personal directory: {os.path.abspath(PERSONAL_DIR)}")
print(f"Database file: {os.path.abspath(DATA_FILE)}")
print(f"Database exists: {os.path.exists(DATA_FILE)}\n")


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


def get_next_personal_folder():
    """Tìm số thứ tự tiếp theo cho folder Personal"""
    existing_folders = [f for f in os.listdir(PERSONAL_DIR) 
                       if os.path.isdir(os.path.join(PERSONAL_DIR, f)) 
                       and f.startswith("Personal")]
    
    if not existing_folders:
        return 1
    
    # Lấy số lớn nhất
    numbers = []
    for folder in existing_folders:
        try:
            num = int(folder.replace("Personal", ""))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1


# Register    
def Face_Register(image_paths, infor, data_file=DATA_FILE):
    """
    image_paths: list các đường dẫn ảnh từ máy local
    infor: thông tin người dùng
    """
    embeddings = []
    register = FaceVerification()
    
    # Tạo folder Personal mới cho người này
    next_num = get_next_personal_folder()
    person_folder = os.path.join(PERSONAL_DIR, f"Personal{next_num}")
    os.makedirs(person_folder, exist_ok=True)
    print(f"Created folder: {person_folder}")
    
    # Copy ảnh từ máy local vào folder Personal
    copied_paths = []
    for i, img_path in enumerate(image_paths):
        # Kiểm tra file có tồn tại
        if not os.path.exists(img_path):
            print(f"Error: File not found: {img_path}")
            return None
        
        # Copy ảnh vào folder Personal
        filename = f"image_{i+1}{os.path.splitext(img_path)[1]}"
        dest_path = os.path.join(person_folder, filename)
        shutil.copy2(img_path, dest_path)
        copied_paths.append(dest_path)
        print(f"Copied: {img_path} -> {dest_path}")
    
    # Đọc và xử lý ảnh
    for i, img_path in enumerate(copied_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Warning: Cannot read image {i+1}, skipping...")
            continue
        
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
        # Xóa folder vì không có face hợp lệ
        shutil.rmtree(person_folder)
        return None
    
    final_embedding = np.mean(embeddings, axis=0)
    
    # Load dữ liệu cũ hoặc tạo mới
    if os.path.exists(data_file):
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded existing database with {len(data)} users")
        except (EOFError, pickle.UnpicklingError):
            print("Warning: data.pkl is corrupted, creating new file...")
            data = {}
    else:
        print("Creating new database file...")
        data = {}
        
    student_id = infor["student_id"]
    data[student_id] = {
        "name": infor["name"],
        "embedding": final_embedding.tolist(),
        "folder": f"Personal{next_num}"
    }
    
    # Lưu file
    with open(data_file, "wb") as f:
        pickle.dump(data, f)
    
    print("Register successful!")
    print(f"Used {len(embeddings)}/{len(image_paths)} images")
    print(f"Total users in database: {len(data)}")
    return final_embedding


# Verification from cam        
def Face_Verification(image, data_file=DATA_FILE, threshold=0.75):
    if image is None:
        print("Read image Error!")
        return None
    
    # Kiểm tra file database có tồn tại không
    if not os.path.exists(data_file):
        print(f"Error: Database file not found at {data_file}")
        print("Please register at least one person first!")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    verifier = FaceVerification()
    
    # Phải detect face trước
    face_input = verifier.detect_and_extract_face(image_rgb)
    
    if face_input is None:
        print("No face detected in frame")
        return None
    
    # Lấy embedding từ face đã detect
    emb_input = verifier.get_embedding(face_input)
    
    if emb_input is None:
        print("Cannot get embedding")
        return None
    
    # Load database
    try:
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found")
        return None
    except EOFError:
        print("Error: Database file is corrupted (empty)")
        return None
    except Exception as e:
        print(f"Load data Error: {e}")
        return None
    
    # Kiểm tra database có dữ liệu không
    if not data or len(data) == 0:
        print("Error: No registered users in database")
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
    print("-" * 60)
    
    while True:
        choice = int(input("Register (1), Verify (2), Out (0) Which choice?: "))
        
        if choice == 1:
            print("Registration Mode")
            print("Please provide 3 image paths from your local machine:\n")
    
            # Nhập đường dẫn ảnh từ máy local
            image_paths = []
            for i in range(1, 4):
                raw_path = input(f"Path {i} (from your computer): ").strip()
                
                if not raw_path:
                    print("Path Error: Path cannot be empty")
                    return None
                
                # Kiểm tra file có tồn tại không
                if not os.path.exists(raw_path):
                    print(f"Error: File not found: {raw_path}")
                    return None
                
                image_paths.append(raw_path)
            
            # Nhập thông tin
            student_id_input = input("Student ID: ").strip()
            name_input = input("Name: ").strip()
            
            if not student_id_input or not name_input:
                print("Error: Student ID and Name cannot be empty")
                return None
    
            info = {
                'student_id': student_id_input,
                'name': name_input
            }

            Face_Register(image_paths, info)
        
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
        print("-" * 60)


if __name__ == "__main__":
    main()