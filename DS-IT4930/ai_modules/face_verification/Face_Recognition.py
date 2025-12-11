import cv2
import numpy as np
import pickle
import os
import shutil
from ai_modules.face_verification.model_loader import load_facenet, load_mtcnn


class FaceRecog:
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.DATA_FILE = os.path.join(self.DATA_DIR, "data.pkl")
        self.STUDENTS_DIR = os.path.join(self.DATA_DIR, "Students")

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.STUDENTS_DIR, exist_ok=True)

        # Load model
        self.facenet = load_facenet()
        self.detector = load_mtcnn() 

    
    # Nhận diện khuôn mặt
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

    def get_embedding(self, face_image):
        if face_image is None:
            return None

        face_image = face_image.astype('float32')
        face_image = np.expand_dims(face_image, axis=0)
        embedding = self.facenet.embeddings(face_image)[0]
        return embedding

    def calculate_distance(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)


    # Đăng ký dữ liệu mới
    def register(self, student_id, name, image_files):
        person_folder = os.path.join(self.STUDENTS_DIR, student_id)
        os.makedirs(person_folder, exist_ok=True)

        embeddings = []
    
        # Lưu và xử lý từng ảnh
        for idx, img_file in enumerate(image_files):
            if img_file is None:
                continue
            
            # Đọc ảnh từ uploaded file
            arr = np.frombuffer(img_file.getvalue(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
            if img is None:
                continue
        
            # Lưu ảnh gốc vào thư mục
            img_path = os.path.join(person_folder, f"{idx+1}.jpg")
            cv2.imwrite(img_path, img)
        
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = self.detect_and_extract_face(img_rgb)

            if face is None:
                print(f"Không phát hiện khuôn mặt trong ảnh {idx+1}")
                continue

            emb = self.get_embedding(face)
            if emb is not None:
                embeddings.append(emb)

        # Kiểm tra có phát hiện được khuôn mặt không
        if len(embeddings) == 0:
            print("Không phát hiện được khuôn mặt hợp lệ trong các ảnh")
            shutil.rmtree(person_folder)
            return None

        final_embedding = np.mean(embeddings, axis=0)

        # Load database (tạo mới nếu chưa có)
        if os.path.exists(self.DATA_FILE):
            try:
                with open(self.DATA_FILE, "rb") as f:
                    data = pickle.load(f)
            except:
                data = {}
        else:
            data = {}

        # Lưu thông tin sinh viên
        data[student_id] = {
            "name": name,
            "embedding": final_embedding.tolist(),
            "folder": student_id
        }

        # Ghi lại vào file
        with open(self.DATA_FILE, "wb") as f:
            pickle.dump(data, f)

        print(f"Đăng ký thành công sinh viên {student_id} - {name}")
        return final_embedding


    # Xác thực khuôn ămtj
    def verify_from_frame(self, frame, threshold=0.75):
        if not os.path.exists(self.DATA_FILE):
            print("Database not found.")
            return None

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = self.detect_and_extract_face(img_rgb)
        if face is None:
            print("No face detected")
            return None

        emb_input = self.get_embedding(face)
        if emb_input is None:
            print("Cannot create embedding")
            return None

        # Load data.pkl
        with open(self.DATA_FILE, "rb") as f:
            data = pickle.load(f)

        best_id = None
        best_name = None
        best_distance = 999

        for student_id, infor in data.items():
            emb_db = np.array(infor["embedding"])
            dist = self.calculate_distance(emb_input, emb_db)

            if dist < best_distance:
                best_distance = dist
                best_id = student_id
                best_name = infor["name"]

        is_match = best_distance < threshold

        return {
            "student_id": best_id,
            "name": best_name,
            "distance": best_distance,
            "match": is_match
        }
