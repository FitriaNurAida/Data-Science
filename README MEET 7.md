# Pertemuan 7 — Model Evaluation 2

Materi pertemuan ketujuh mata kuliah Data Science: cross-validation, tuning, memilih batas peluang berdasarkan untung, dan simulasi biaya-manfaat kampanye.

| File | Isi |
|---|---|
| `README MEET 7.md` | Metadata dataset, langkah, dan hasil (dokumen ini) |
| `07_Model_Evaluation_2.pptx` | Slide materi (20 slide, dengan catatan pembicara) |
| `07_Model_Evaluation_2.ipynb` | Notebook praktik untuk Google Colab |
| `m7_01` … `m7_05` `.png` | Ilustrasi alur kode untuk blok-blok utama notebook |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-07/
├── README.md
├── 07_Model_Evaluation_2.pptx
└── 07_Model_Evaluation_2.ipynb
```

---

## Dataset

Dataset yang sama dengan Pertemuan 2–6: Customer Personality Analysis. Dalam business case, perusahaan ini disebut **RetailKu** (nama fiktif).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle, diunggah oleh Akash Patel (`imakash3011`) |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| File masukan | `marketing_segmen.csv`, hasil Pertemuan 4 (2.026 pelanggan) |
| Lokasi bawaan di Drive | `UNIPA 2026/Meet 7/` (variabel `FOLDER`) |
| Target | `Response` (15,3% merespons) |
| Biaya dan untung | Biaya per katalog **3**, keuntungan per respons **11** |

**Tentang biaya dan untung.** Dataset mentah punya kolom `Z_CostContact` (selalu 3) dan `Z_Revenue` (selalu 11) yang dibuang di Pertemuan 2 karena isinya konstan. Dokumentasi dataset tidak menjelaskan artinya; di cerita kelas, keduanya dipakai sebagai biaya kontak dan keuntungan per respons, mengikuti tafsiran yang umum di komunitas Kaggle. Satuan mata uangnya tidak disebutkan.

Fitur, pipeline persiapan, dan pembagian data (80/20, `stratify`, `random_state=42`) sama dengan Pertemuan 5 dan 6.

---

## Hasil

### 1. Satu split bisa menipu

Logistic Regression yang sama dilatih dengan 30 `random_state` berbeda:

| Ukuran | Terendah | Rata-rata | Tertinggi |
|---|---|---|---|
| Akurasi | 0,855 | 0,881 | 0,901 |
| F1 | 0,366 | 0,501 | 0,602 |
| ROC-AUC | 0,832 | 0,874 | 0,929 |

### 2. Cross-validation (5-fold, data latih 1.620 pelanggan)

AUC per fold Logistic Regression: 0,873 · 0,859 · 0,850 · 0,881 · 0,867 → **0,866 ± 0,011**.

| Model | AUC | ± sd | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0,866 | 0,011 | 0,505 | 0,719 | 0,391 |
| LogReg `class_weight="balanced"` | 0,866 | 0,011 | 0,522 | 0,398 | 0,758 |
| Decision Tree (depth 3) | 0,750 | 0,031 | 0,352 | 0,596 | 0,258 |
| Random Forest (200 pohon) | 0,848 | 0,019 | 0,450 | 0,686 | 0,339 |

### 3. Tuning

- Kurva validasi Decision Tree: AUC CV tertinggi di `max_depth=4` (0,773); fold latih terus naik sampai 1,0.
- `GridSearchCV` Decision Tree (21 kombinasi × 5 fold): terbaik `max_depth=6, min_samples_leaf=20`, AUC CV 0,806. Tanpa `min_samples_leaf=20`, depth 6 hanya 0,694.

| Model | Hyperparameter terbaik | AUC CV | AUC uji |
|---|---|---|---|
| Decision Tree | `max_depth=6, min_samples_leaf=20` | 0,806 | 0,810 |
| **Logistic Regression** | `C=0.1` | **0,867** | **0,875** |
| Random Forest | `max_depth=8, min_samples_leaf=1` | 0,859 | 0,848 |

Logistic Regression hasil tuning dipilih sebagai model kampanye. Tuning `C` hanya menaikkan AUC CV dari 0,866 ke 0,867: tuning bukan sihir.

### 4. Batas peluang berbasis untung

`untung = 11 × TP − 3 × (TP + FP)`. Titik impas: precision ≥ 3/11 ≈ 27%.

Batas dipilih dengan `cross_val_predict` di data latih → **batas terbaik 0,21**. Hasil di data uji (406 pelanggan, 62 responden):

| Strategi | Dikirim | Responden | Precision | Untung |
|---|---|---|---|---|
| Tidak mengirim | 0 | 0 | — | 0 |
| Kirim ke semua | 406 | 62 | 15% | **−536** |
| Model, batas 0,5 | 32 | 22 | 69% | 146 |
| **Model, batas 0,21** | 94 | 43 | 46% | **191** |

### 5. Simulasi skenario

| Biaya | Untung per respons | Titik impas | Batas terbaik | Dikirim (uji) | Untung model | Untung kirim semua |
|---|---|---|---|---|---|---|
| 3 | 11 | 27% | 0,21 | 94 | 191 | −536 |
| 1 | 11 | 9% | 0,11 | 152 | 431 | 276 |
| 6 | 11 | 55% | 0,59 | 25 | 59 | −1.754 |
| 3 | 20 | 15% | 0,12 | 149 | 613 | 22 |
| 3 | 6 | 50% | 0,42 | 42 | 36 | −846 |

### 6. Model final dan daftar kirim

Model dilatih ulang dengan semua 2.026 pelanggan dan disimpan bersama batasnya. Dengan batas 0,21, **459 dari 2.026 pelanggan** dikirimi katalog: Premium 39%, Keluarga Mapan 26%, Pemburu Diskon 26%, Hemat 10%.

### 7. Cross-validation regresi (`TotalSpend`, fitur profil)

| Model | MAE per fold | Rata-rata |
|---|---|---|
| Linear Regression | 232 · 232 · 242 · 246 · 243 | 239 ± 6 |
| Random Forest | 180 · 175 · 203 · 218 · 180 | 191 ± 17 |

Angka bisa sedikit berbeda jika versi scikit-learn berbeda.

### File keluaran (disimpan ke `FOLDER`)

| File | Isi |
|---|---|
| `model_kampanye7.joblib` | Dict berisi `model`, `batas`, `fitur`, `biaya_kontak`, `untung_respons` |
| `daftar_kirim_kampanye7.csv` | `ID`, `NamaSegmen`, `peluang`, `kirim_katalog` untuk 2.026 pelanggan, urut dari peluang tertinggi |

---

## Tentang materi pertemuan ini

**Business case.** Masih Juli 2014. Kampanye ke-7 dikirim lewat katalog cetak. Direktur keuangan: *"Setiap katalog memakan biaya 3. Setiap pelanggan yang merespons memberi keuntungan 11. Kirim ke siapa supaya untungnya paling besar? Dan apakah angka kalian bukan kebetulan?"*

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Apakah hasil model kita kebetulan? | Cross-validation |
| Model dan pengaturan mana yang terbaik? | Tuning hyperparameter dengan `GridSearchCV` |
| Pelanggan mana yang dikirimi katalog? | Batas peluang dengan untung terbesar |
| Bagaimana jika biaya atau keuntungan berubah? | Simulasi skenario |
| Bagaimana model dipakai besok? | Menyimpan model final dan membuat daftar kirim |

**Alur notebook.** 30 pembagian data → ide k-fold → `cross_val_score` dengan pipeline → `cross_validate` empat model → parameter vs hyperparameter → kurva validasi → `GridSearchCV` (pohon, LogReg, Random Forest) → data uji sekali → fungsi untung dan titik impas → `cross_val_predict` dan kurva untung → strategi di data uji → simulasi skenario → model final, `joblib`, daftar kirim → CV untuk regresi.

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Menjelaskan kenapa satu kali pembagian data latih-uji bisa menipu.
2. Melakukan k-fold cross-validation dengan pipeline, tanpa kebocoran data.
3. Membedakan parameter dan hyperparameter, lalu melakukan tuning dengan `GridSearchCV`.
4. Menjaga data uji tetap bersih: dipakai sekali di akhir.
5. Memilih batas peluang berdasarkan biaya dan keuntungan.
6. Menjalankan simulasi skenario saat biaya atau keuntungan berubah.
7. Menyimpan model final dan membuat daftar pelanggan untuk kampanye.

**Cara memakai notebook.** Buka `07_Model_Evaluation_2.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana, dan pastikan `marketing_segmen.csv` ada di folder `FOLDER` di Google Drive. Seluruh notebook berjalan ±1–2 menit di Colab (paling lama: `GridSearchCV` Random Forest).

**Perkiraan waktu.** Slide ±40 menit, praktik notebook 75–90 menit.

**Output mahasiswa.** Notebook yang sudah dijalankan, `daftar_kirim_kampanye7.csv`, dan memo 5–7 kalimat untuk direksi.

**Selanjutnya.** Final project dengan `retailku_final_project.csv`: dikumpulkan di Pertemuan 9, presentasi Pertemuan 9 dan 10.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
