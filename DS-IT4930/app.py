import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os
import numpy as np
import cv2
import sys


# Đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

face_data_dir = os.path.join(project_root, "ai_modules", "face_verification", "data")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model
from ai_modules.face_verification.Face_Recognition import FaceRecog
face_recog = FaceRecog(data_dir=face_data_dir)

st.title("Giám Sát Thi Cử")

# Session state
if "verified" not in st.session_state:
    st.session_state.verified = False

if "exam_started" not in st.session_state:
    st.session_state.exam_started = False

if "need_re_register" not in st.session_state:
    st.session_state.need_re_register = False

# Sidebar
menu = st.sidebar.radio("Chọn chế độ:", ["Xác thực", "Đăng ký lại dữ liệu"])

# Bắt buộc xác thực
if menu == "Xác thực":
    if not st.session_state.verified:
        st.subheader("Step 1: Xác thực khuôn mặt sinh viên")

        img = st.camera_input("Chụp ảnh để xác thực")

        if img is not None:
            file_bytes = np.frombuffer(img.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            result = face_recog.verify_from_frame(frame)
            if result is None:
                st.error("Không phát hiện được khuôn mặt. Vui lòng chụp rõ hơn.")
            else:
                if result["match"]:
                    st.success("XÁC THỰC THÀNH CÔNG!")

                    st.write("Thông tin sinh viên:")
                    st.write(f"Họ tên: {result.get('name', 'Không có dữ liệu')}")
                    st.write(f"MSSV: {result.get('student_id', 'Không có dữ liệu')}")

                    st.session_state.verified = True
                else:
                    st.error("Không trùng khớp với dữ liệu hệ thống.")

                    st.warning("Có thể dữ liệu trước đó bị lỗi hoặc không đúng.")
                    st.info("Liên hệ giám thị để đăng ký lại dữ liệu khuôn mặt.")

                    st.session_state.need_re_register = True

                    if st.button("Yêu cầu đăng ký lại dữ liệu"):
                        st.switch_page("app.py")

    # kết thúc xác thực vào thi
    elif not st.session_state.exam_started:
        st.subheader("Step 2: Vào làm bài thi")

        if st.button("Vào làm bài"):
            st.session_state.exam_started = True

    else:
        st.success("Bạn đang trong bài thi!")
        st.write("... Giao diện bài thi ở đây ...")


# Đăng ký khuôn mặt
elif menu == "Đăng ký lại dữ liệu":
    st.header("Đăng ký lại dữ liệu khuôn mặt (Dành cho Giám thị)")

    img1 = st.file_uploader("Ảnh 1:", type=["jpg", "jpeg", "png"], key="img1")
    img2 = st.file_uploader("Ảnh 2:", type=["jpg", "jpeg", "png"], key="img2")
    img3 = st.file_uploader("Ảnh 3:", type=["jpg", "jpeg", "png"], key="img3")

    student_id = st.text_input("MSSV")
    name = st.text_input("Họ tên")

    if st.button("Lưu dữ liệu mới"):
        if not student_id or not name:
            st.error("Vui lòng nhập đầy đủ MSSV và Họ tên!")
        elif not img1 or not img2 or not img3:
            st.error("Vui lòng upload đủ 3 ảnh!")
        else:
            try:
                # Gọi hàm register với danh sách ảnh
                uploaded_imgs = [img1, img2, img3]
                
                result = face_recog.register(
                    student_id=student_id,
                    name=name,
                    image_files=uploaded_imgs
                )
                
                if result is not None:
                    st.success(f"Đăng ký dữ liệu khuôn mặt thành công cho {name} (MSSV: {student_id})!")
                else:
                    st.error("Không thể đăng ký! Vui lòng kiểm tra lại các ảnh có chứa khuôn mặt rõ ràng không.")
                    
            except Exception as e:
                st.error(f"Lỗi khi đăng ký: {str(e)}")