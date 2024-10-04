import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return image
    face = faces[0]
    landmarks = predictor(gray, face)
    landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]
    src_points = np.array(landmarks_points[36:48], dtype="float32")
    dst_points = np.array([(30, 40), (60, 40), (45, 70)], dtype="float32")
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned_image
