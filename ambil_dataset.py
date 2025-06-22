import cv2
import os

# Buat folder dataset jika belum ada
DATASET_DIR = 'pose_images'
os.makedirs(DATASET_DIR, exist_ok=True)

# Input nama label
label = input("Masukkan label pose (contoh: maju, mundur, kanan, dsb): ").strip().lower()
save_dir = os.path.join(DATASET_DIR, label)
os.makedirs(save_dir, exist_ok=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
img_count = len(os.listdir(save_dir))  # Lanjutkan dari nomor terakhir jika folder sudah ada

print("\nTekan [Spasi] untuk simpan gambar, [q] untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
        #mirror kamera
    frame = cv2.flip(frame, 1)  # Mirror horizontalmaju
        #atur cahaya
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)



    # Tampilkan frame tanpa tulisan label
    cv2.imshow("Ambil Dataset Pose", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # Spasi untuk simpan
        img_path = os.path.join(save_dir, f"{label}_{img_count:03d}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[+] Gambar disimpan: {img_path}")
        img_count += 1

    elif key == ord('q'):  # q untuk keluar
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()