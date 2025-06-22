import os
import cv2
import mediapipe as mp
import csv

# Path ke folder gambar hover (isi hanya gambar untuk satu label pose)
folder_path = r'C:\Users\acer\Documents\elmobot\pose_images\kanan'
output_csv = 'kanan_pose.csv'  # Nama file hasil CSV

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Gagal membaca {filename}")
            continue

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print(f"⚠️ Tidak ada pose di {filename}")
            continue

        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
            keypoints.append(landmark.visibility)

        # Tambahkan label di akhir (ganti sesuai label pose yang benar)
        keypoints.append("hover")

        data.append(keypoints)
        print(f"✅ Pose dari {filename} berhasil diekstrak.")

# Simpan ke CSV
header = []
for i in range(33):  # total keypoints
    header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
header.append('label')

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f'\n✅ Semua data disimpan ke {output_csv}')