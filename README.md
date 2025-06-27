# ✨ PoseEstimation Using Mediapipe RandomForest

[![Python](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)


> Proyek ini menggunakan MediaPipe dan RandomForest untuk estimasi pose.

## ✨ Fitur Utama

* **Pengambilan Data:** Skrip `ambil_dataset.py` kemungkinan digunakan untuk mengambil dataset gambar untuk pelatihan model estimasi pose.
* **Pembuatan Dataset:** Skrip `buat_dataset_perpose.py` memproses dan mungkin mentransformasikan data mentah menjadi format yang sesuai untuk pelatihan model RandomForest.
* **Penggabungan Dataset:** Skrip `gabungpose_csv.py` kemungkinan menggabungkan beberapa file dataset menjadi satu dataset yang koheren untuk pelatihan.
* **Pelatihan Model:** Skrip `training_dataset_baru.py` melatih model RandomForest untuk estimasi pose berdasarkan dataset yang telah disiapkan.


## 🛠️ Tumpukan Teknologi

| Kategori        | Teknologi | Catatan                                      |
|-----------------|------------|----------------------------------------------|
| Bahasa Pemrograman | Python     | Bahasa utama untuk pengembangan proyek.      |
| Library          | MediaPipe  | Digunakan untuk ekstraksi fitur pose.          |
| Library          | scikit-learn (diprediksi) | Kemungkinan digunakan untuk implementasi RandomForest. |


## 🏛️ Tinjauan Arsitektur

Arsitektur proyek ini didasarkan pada alur pemrosesan data yang sederhana. Pertama, data dikumpulkan dan diproses (`ambil_dataset.py`, `buat_dataset_perpose.py`). Kemudian, beberapa dataset digabungkan (`gabungpose_csv.py`). Terakhir, model RandomForest dilatih menggunakan dataset yang telah disiapkan (`training_dataset_baru.py`).  Tidak ada informasi lebih lanjut yang tersedia mengenai detail arsitektur internal.


## 🚀 Memulai

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/HinXs/PoseEstimation-using-Mediapipe-RandomForest.git
   cd PoseEstimation-using-Mediapipe-RandomForest
   ```
2. **Instalasi Dependensi (Jika diperlukan):**  Informasi manajer paket tidak tersedia, sehingga instalasi dependensi harus dilakukan secara manual dengan memasang library yang dibutuhkan seperti MediaPipe dan scikit-learn.
3. **Menjalankan Skrip:** Jalankan skrip Python sesuai kebutuhan, misalnya:
   ```bash
   python ambil_dataset.py
   python buat_dataset_perpose.py
   python gabungpose_csv.py
   python training_dataset_baru.py
   ```


## 📂 Struktur File

```
/
├── README.md
├── ambil_dataset.py
├── buat_dataset_perpose.py
├── gabungpose_csv.py
└── training_dataset_baru.py
```

* **Root Directory (`/`):** Berisi semua file dan skrip proyek.


