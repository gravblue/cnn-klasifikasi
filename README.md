# Skin Condition Classification 

Proyek klasifikasi gambar untuk mendeteksi kondisi kulit (Acne, Eczema, Psoriasis, dan Normal Skin) menggunakan transfer learning dengan **EfficientNetB3**.

## 📌 Overview

Model CNN berbasis EfficientNetB3 (fine-tuned penuh) dilatih untuk mengklasifikasikan gambar kulit ke dalam 4 kelas. Setelah 10 epoch training, model mencapai akurasi validasi 95,16% dengan performa terbaik pada kelas *Normal Skin* dan *Acne*.

## 📂 Dataset

Dataset diambil dari Kaggle: [`acne-psoriasis-eczema-vs-all-skin-diseases`](https://www.kaggle.com/datasets/sufiahmad883/acne-psoriasis-eczema-vs-all-skin-diseases)

| Kelas | Jumlah Gambar |
|---|---|
| Acne | 4.198 |
| Psoriasis | 3.812 |
| Eczema | 3.200 |
| Normal Skin | 2.158 |
| **Total** | **13.368** |

Split dataset: **70% train / 15% validation / 15% test** (stratified berdasarkan kelas).

## 🧠 Arsitektur Model

- **Base model:** EfficientNetB3 (pretrained ImageNet, full fine-tuning — semua layer trainable)
- **Input size:** 300×300×3
- **Head:** Conv2D(256) → BatchNorm → Conv2D(128) → BatchNorm → GlobalAveragePooling2D → Dense(256) → BatchNorm → Dropout(0.6) → Dense(128) → BatchNorm → Dropout(0.6) → Dense(4, softmax)
- **Total parameter:** 14.687.283 (14.598.444 trainable)
- **Optimizer:** Adam (lr=1e-4)
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping (patience=6, monitor val_accuracy) & ReduceLROnPlateau (factor=0.5, patience=2)
- **Class weighting:** balanced, untuk menangani ketidakseimbangan jumlah data antar kelas

### Data Augmentation (train set)
Rotation ±20°, width/height shift 0.1, shear 0.1, zoom 0.1, horizontal flip.

## 📊 Hasil

Training 10 epoch:

| Metric | Train | Validation |
|---|---|---|
| Accuracy | 95,20% | 95,16% |
| Loss | 0,1326 | 0,1691 |

**Classification Report (Validation Set):**

| Kelas | Precision | Recall | F1-score |
|---|---|---|---|
| Acne | 0.98 | 0.97 | 0.97 |
| Eczema | 0.91 | 0.94 | 0.92 |
| Normal Skin | 0.99 | 0.99 | 0.99 |
| Psoriasis | 0.93 | 0.92 | 0.93 |
| Accuracy | | | 0.95 |

Gap antara train dan validation accuracy kecil, menunjukkan model belajar dengan baik tanpa overfitting signifikan.

## 🛠️ Tech Stack

- TensorFlow / Keras (EfficientNetB3, ImageDataGenerator)
- scikit-learn (train_test_split, class_weight, classification_report, confusion_matrix)
- Pandas, NumPy
- Matplotlib, Seaborn (visualisasi)
- Kaggle API (download dataset)

## 📁 Struktur Notebook

1. **Import Library**:load semua dependency
2. **Data Preparation**: download dataset dari Kaggle, eksplorasi jumlah & sampel gambar per kelas
3. **Data Preprocessing**: split train/val/test, `ImageDataGenerator` + augmentasi
4. **Modelling**: bangun & training model EfficientNetB3
5. **Evaluasi & Visualisasi**: classification report, confusion matrix, plot accuracy/loss
