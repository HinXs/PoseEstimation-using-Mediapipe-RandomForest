import pandas as pd
import glob
import os

folder_path = r'C:\Users\acer\Documents\elmobot\dataset_csv'
csv_files = glob.glob(folder_path + r'\*_pose.csv')

print(f"📂 Jumlah file ditemukan: {len(csv_files)}")
all_dataframes = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    # Ambil nama label dari nama file, contoh: kiri_pose.csv -> kiri
    label_name = os.path.basename(file_path).replace('_pose.csv', '')
    
    # Ganti semua nilai kolom 'label' dengan nama yang benar
    df['label'] = label_name
    print(f"📄 {file_path} -> {df.shape[0]} data, label diubah jadi: '{label_name}'")
    
    all_dataframes.append(df)

# Gabungkan semua
all_data = pd.concat(all_dataframes, ignore_index=True)

# Simpan ke file baru
output_file = folder_path + r'\pose_dataset_rf.csv'
all_data.to_csv(output_file, index=False)

print(f"\n✅ Dataset akhir disimpan ke: {output_file}")
