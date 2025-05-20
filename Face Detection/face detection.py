import cv2
import os
import numpy as np
import face_recognition

# 加载已知同学的照片
known_faces_dir = "known_faces"
known_faces = []
known_names = []
for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# 加载待识别的照片
unknown_faces_dir = "unknown_faces"
for filename in os.listdir(unknown_faces_dir):
    image = face_recognition.load_image_file(os.path.join(unknown_faces_dir, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # 对每张待识别的照片进行人脸匹配
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # 找到最相似的已知同学
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # 在照片中标记出人脸并显示识别结果
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)
