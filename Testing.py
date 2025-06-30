import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
from sklearn.metrics import classification_report, accuracy_score

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)

# Fungsi untuk memproses frame dan ekstrak landmark pose
def extract_pose_landmarks(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    return np.array(landmarks) if len(landmarks) > 0 else None

# Fungsi untuk memuat model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Fungsi utama untuk testing
def test_pose_estimation(model_path, test_video_path=None, camera_index=0):
    # Memuat model Random Forest
    model = load_model(model_path)
    
    # Inisialisasi video capture
    if test_video_path:
        cap = cv2.VideoCapture(test_video_path)
    else:
        cap = cv2.VideoCapture(camera_index)
    
    # Variabel untuk menyimpan hasil prediksi (jika ada ground truth)
    predictions = []
    true_labels = []  # Diisi jika Anda memiliki ground truth
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ekstrak landmark pose
        landmarks = extract_pose_landmarks(frame)
        
        if landmarks is not None:
            # Reshape landmarks untuk prediksi (1 sampel, n fitur)
            landmarks = landmarks.reshape(1, -1)
            
            # Prediksi dengan Random Forest
            prediction = model.predict(landmarks)
            predictions.append(prediction[0])
            
            # Jika Anda memiliki ground truth, tambahkan ke true_labels
            # true_labels.append(true_label_for_frame)
            
            # Visualisasi pose dan hasil prediksi
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Tampilkan prediksi di frame
            cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Tampilkan frame
        cv2.imshow('Pose Estimation Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Evaluasi performa jika ada ground truth
    if len(true_labels) > 0:
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.2f}")
        print(classification_report(true_labels, predictions))
    
    # Rilis resources
    cap.release()
    cv2.destroyAllWindows()

# Jika Anda ingin menguji dengan dataset yang sudah ada
def test_with_dataset(model_path, X_test, y_test):
    model = load_model(model_path)
    
    # Prediksi pada test set
    y_pred = model.predict(X_test)
    
    # Evaluasi model
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    return y_pred

# Contoh penggunaan
if __name__ == "__main__":
    # Path ke model yang sudah disimpan
    MODEL_PATH = 'pose_random_forest_model.pkl'
    
    # Untuk testing dengan webcam
    print("Testing with webcam...")
    test_pose_estimation(MODEL_PATH)
    
    # Untuk testing dengan video file
    # test_pose_estimation(MODEL_PATH, test_video_path='test_video.mp4')
    
    # Untuk testing dengan dataset (jika ada)
    # X_test dan y_test harus sudah di-load sebelumnya
    # test_with_dataset(MODEL_PATH, X_test, y_test)
