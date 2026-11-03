# Pertemuan 5 — Classification & Regression

Materi pertemuan kelima mata kuliah Data Science: supervised learning untuk klasifikasi dan regresi.

| File | Isi |
|---|---|
| `README MEET 5.md` | Metadata dataset, fitur, model, dan hasil (dokumen ini) |
| `05_Classification_Regression.pptx` | Slide materi (22 slide, dengan catatan pembicara) |
| `05_Classification_Regression.ipynb` | Notebook praktik untuk Google Colab |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-05/
├── README.md
├── 05_Classification_Regression.pptx
└── 05_Classification_Regression.ipynb
```

---

## Dataset: Customer Personality Analysis

Dataset yang sama dengan Pertemuan 2–4: data pelanggan sebuah perusahaan ritel yang menjual kebutuhan rumah tangga lewat toko, situs web, dan katalog. Satu baris adalah satu pelanggan. Dalam business case, perusahaan ini disebut **RetailKu** (nama fiktif).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle, diunggah oleh Akash Patel (`imakash3011`) |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| File masukan | `marketing_segmen.csv`, hasil Pertemuan 4 (data bersih + fitur + kolom `Segmen` dan `NamaSegmen`) |
| Jumlah baris | 2.026 pelanggan |
| Target klasifikasi | `Response`: 1 jika pelanggan menerima tawaran pada kampanye terakhir (15,3% pelanggan) |
| Target regresi | `TotalSpend`: total belanja selama dua tahun (rata-rata 607, median 397) |

Penjelasan kolom asli ada di `README MEET 2.md`; fitur baru di `README MEET 3.md`; segmen di `README MEET 4.md`.

### Membaca data

```python
import pandas as pd

df = pd.read_csv("marketing_segmen.csv")   # file keluaran Pertemuan 4
```

---

## Klasifikasi: memprediksi `Response`

**Fitur dan pipeline** sama dengan Pertemuan 3: 12 fitur angka (`Income`, `Age`, `Tenure_days`, `Recency`, `Children`, `TotalSpend_log`, `TotalPurchases`, `AvgSpendPerPurchase`, `NumDealsPurchases`, `NumWebVisitsMonth`, `TotalAcceptedPrev`, `Complain`), `Education` (ordinal), dan `Marital_Status` (one-hot). Model berbasis pohon memakai pipeline yang sama tanpa scaling.

**Pembagian data:** 80% latih (1.620) dan 20% uji (406, berisi 62 responden), `stratify=y`, `random_state=42`.

| Model | Akurasi uji | Responden tertangkap | Akurasi latih |
|---|---|---|---|
| Model malas (`DummyClassifier`) | 84,7% | 0 dari 62 (0%) | 84,7% |
| Logistic Regression | 87,7% | 24 dari 62 (39%) | 88,8% |
| Decision Tree (`max_depth=3`) | 86,0% | 22 dari 62 (35%) | 88,2% |
| KNN (`n_neighbors=5`) | 86,0% | 18 dari 62 (29%) | 90,0% |
| Random Forest (300 pohon) | 86,5% | 20 dari 62 (32%) | 99,5% |

**Temuan penting untuk diskusi:**

- Model malas 84,7% akurat tetapi tidak menangkap satu pun responden: akurasi menipu pada kelas tidak seimbang.
- Decision Tree tanpa batas kedalaman mencapai 99,5% di data latih tetapi hanya sekitar 82% di data uji (overfitting).
- Fitur paling berpengaruh pada Logistic Regression: `TotalAcceptedPrev` (+), `Tenure_days` (+), `TotalSpend_log` (+), `Recency` (−).
- Jika penawaran dikirim ke 20% pelanggan dengan peluang tertinggi (81 dari 406), model menangkap **41 dari 62 responden (66%)**, dibanding sekitar 12 jika dipilih acak. Tingkat respons kelompok ini 51%.

## Regresi: memprediksi `TotalSpend`

**Fitur:** hanya profil yang diketahui saat pelanggan mendaftar: `Income`, `Age`, `Children`, `Tenure_days`, `Education`, `Marital_Status`.

**Tidak boleh dipakai** karena dihitung dari target (kebocoran): kolom `Mnt...`, `TotalSpend_log`, `AvgSpendPerPurchase`, dan `Segmen` dari Pertemuan 4.

**Pembagian data:** 80/20, `random_state=42` (tanpa `stratify` karena target berupa angka).

| Model | MAE uji | R² uji | R² latih |
|---|---|---|---|
| Model malas (selalu rata-rata) | 524 | 0,00 | 0,00 |
| Linear Regression | 237 | 0,76 | 0,70 |
| Decision Tree (`max_depth=4`) | 211 | 0,76 | 0,79 |
| Random Forest (300 pohon, `min_samples_leaf=5`) | 186 | 0,81 | 0,89 |

Koefisien Linear Regression dalam satuan asli: +10.000 pendapatan → +190; +1 anak → −210; +1 tahun berlangganan → +185; +1 tahun umur → sekitar +1. Linear Regression bisa menghasilkan tebakan negatif untuk pelanggan berpendapatan rendah dengan banyak anak.

Angka bisa sedikit berbeda jika versi scikit-learn atau hasil pembersihan berbeda.

### File keluaran

`prediksi_response_uji.csv`: 406 baris data uji berisi `ID`, `Response`, peluang dari Logistic Regression dan Random Forest, serta tebakan 0/1 dari Logistic Regression, Decision Tree, dan Random Forest. File ini menjadi titik awal Pertemuan 6.

---

## Tentang materi pertemuan ini

**Business case.** Masih Juli 2014. Segmen pelanggan dari Pertemuan 4 sudah diterima. Manajer marketing bertanya: *"Anggaran kampanye ke-7 hanya cukup untuk sebagian pelanggan. Siapa yang kemungkinan besar merespons? Dan bisakah kita memperkirakan nilai belanja pelanggan yang baru mendaftar?"*

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Siapa yang akan merespons kampanye ke-7? | Klasifikasi dengan target `Response` |
| Apakah model kita benar-benar berguna? | Baseline, akurasi, confusion matrix |
| Model mana yang paling cocok? | Membandingkan empat model klasifikasi |
| Kenapa model hebat di latihan tapi gagal di ujian? | Overfitting |
| Kepada siapa penawaran dikirim lebih dulu? | Mengurutkan berdasarkan peluang (`predict_proba`) |
| Berapa potensi belanja pelanggan baru? | Regresi dengan target `TotalSpend`, MAE dan R² |

**Alur notebook.** Mengenal target → baseline dan confusion matrix → Logistic Regression, Decision Tree, KNN, Random Forest → perbandingan → kurva overfitting → koefisien dan daftar prioritas → regresi (kebocoran, baseline, Linear Regression, pohon, Random Forest) → perkiraan untuk pelanggan baru → menyimpan prediksi.

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Menjelaskan alur supervised learning: fitur (X), target (y), `fit`, dan `predict`.
2. Membedakan klasifikasi dan regresi dari jenis targetnya.
3. Membuat baseline dan menjelaskan kenapa akurasi tinggi bisa menipu.
4. Melatih dan membandingkan Logistic Regression, Decision Tree, KNN, dan Random Forest.
5. Mengenali overfitting dari selisih skor data latih dan data uji.
6. Membaca model: koefisien, aturan pohon, dan peluang.
7. Melatih model regresi, mencegah kebocoran dari target, dan menilainya dengan MAE dan R².

**Cara memakai notebook.** Buka `05_Classification_Regression.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana, dan sesuaikan `FILE_DATA` dengan lokasi `marketing_segmen.csv` di Google Drive (bawaan: `UNIPA 2026/Meet 5/`).

**Perkiraan waktu.** Slide ±40 menit, praktik notebook 70–80 menit.

**Output mahasiswa.** Notebook yang sudah dijalankan, file `prediksi_response_uji.csv`, tabel rekomendasi model, dan memo 4–6 kalimat untuk manajer marketing.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
