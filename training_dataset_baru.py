import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv('pose_dataset_rf.csv')

# Pisahkan fitur dan label
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("🎯 Akurasi:", accuracy_score(y_test, y_pred))
print("\n📊 Laporan klasifikasi:")
print(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, 'rf_pose_model.pkl')
print("✅ Model Random Forest disimpan sebagai 'rf_pose_model.pkl'")
