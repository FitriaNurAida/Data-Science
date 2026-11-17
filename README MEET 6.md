# Pertemuan 6 — Model Evaluation

Materi pertemuan keenam mata kuliah Data Science: menilai model klasifikasi dan regresi dengan ukuran yang sesuai kebutuhan bisnis.

| File | Isi |
|---|---|
| `README MEET 6.md` | Metadata dataset, ukuran evaluasi, dan hasil (dokumen ini) |
| `06_Model_Evaluation.pptx` | Slide materi (22 slide, dengan catatan pembicara) |
| `06_Model_Evaluation.ipynb` | Notebook praktik untuk Google Colab |
| `m6_01` … `m6_06` `.png` | Ilustrasi alur kode untuk blok-blok utama notebook |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-06/
├── README.md
├── 06_Model_Evaluation.pptx
└── 06_Model_Evaluation.ipynb
```

---

## Dataset

Dataset yang sama dengan Pertemuan 2–5: Customer Personality Analysis. Dalam business case, perusahaan ini disebut **RetailKu** (nama fiktif).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle, diunggah oleh Akash Patel (`imakash3011`) |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| File masukan 1 | `prediksi_response_uji.csv`, hasil Pertemuan 5: 406 pelanggan uji berisi `ID`, `Response`, `peluang_logreg`, `peluang_hutan`, `tebakan_logreg`, `tebakan_pohon`, `tebakan_hutan` |
| File masukan 2 | `marketing_segmen.csv`, hasil Pertemuan 4 (untuk melatih ulang model `class_weight` dan model regresi) |
| Lokasi bawaan di Drive | `UNIPA 2026/Meet 6/` (variabel `FOLDER`) |

Bagian klasifikasi dihitung langsung dari `prediksi_response_uji.csv` **tanpa melatih ulang**. Notebook memeriksa bahwa data uji yang dibuat ulang sama persis dengan Pertemuan 5 (`ID` cocok).

---

## Klasifikasi: memprediksi `Response`

Data uji: 406 pelanggan, 62 merespons (15,3%).

### Confusion matrix Logistic Regression (batas 0,5)

| | Tebak 0 | Tebak 1 |
|---|---|---|
| **Sebenarnya 0** | TN = 332 | FP = 12 |
| **Sebenarnya 1** | FN = 38 | TP = 24 |

### Empat ukuran per model

| Model | Akurasi | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---|---|---|---|---|---|
| Model malas (semua 0) | 0,847 | 0,000 | 0,000 | 0,000 | — | — |
| Logistic Regression | 0,877 | 0,667 | 0,387 | 0,490 | 0,875 | 0,598 |
| Decision Tree (depth 3) | 0,860 | 0,564 | 0,355 | 0,436 | — | — |
| Random Forest | 0,865 | 0,606 | 0,323 | 0,421 | 0,847 | 0,503 |
| LogReg `class_weight="balanced"` | 0,796 | 0,409 | 0,758 | 0,531 | 0,879 | — |

### Temuan penting untuk diskusi

- **Akurasi turun, model lebih berguna.** Dengan `class_weight="balanced"`, akurasi turun dari 87,7% ke 79,6%, tetapi responden tertangkap naik dari 24 ke 47 (recall 0,39 → 0,76), dengan 56 penawaran terbuang tambahan.
- **F1 dan rata-rata harmonik.** Jika semua pelanggan ditebak merespons: precision 0,153, recall 1,000, rata-rata biasa 0,576, tetapi F1 hanya 0,265.
- **Batas peluang** (Logistic Regression): batas 0,2 → precision 0,45 dan recall 0,73; batas 0,5 → 0,67 dan 0,39; batas 0,8 → 0,91 dan 0,16. Memilih batas terbaik dibahas di Pertemuan 7.
- **Arti AUC.** AUC 0,875 = dari 62 × 344 = 21.328 pasangan (responden, non-responden), model memberi skor lebih tinggi kepada responden pada 87,5% pasangan. Notebook membuktikannya dengan menghitung semua pasangan.
- **Kurva gain** (LogReg): menghubungi 10% pelanggan teratas menangkap 44% responden; 20% → 66%; 30% → 79%; 50% → 94%.
- **Memilih ukuran:** kampanye email (murah) → utamakan recall; kampanye katalog + hadiah (mahal) → utamakan precision.

## Regresi: memprediksi `TotalSpend`

Fitur profil saja (`Income`, `Age`, `Children`, `Tenure_days`, `Education`, `Marital_Status`), split 80/20 dengan `random_state=42`, sama seperti Pertemuan 5.

| Model | MAE | RMSE | R² | MAPE | Tebakan negatif |
|---|---|---|---|---|---|
| Model malas (rata-rata) | 524 | 613 | 0,00 | 543% | 0 |
| Linear Regression | 237 | 304 | 0,75 | 182% | 39 |
| Random Forest | 186 | 270 | 0,81 | 84% | 0 |

- **Jebakan MAPE:** pelanggan dengan belanja 8 ditebak −303 dihitung meleset 3.894%. MAPE Linear Regression 182% untuk semua pelanggan, tetapi 54% untuk pelanggan dengan belanja ≥ 100. Ada 127 dari 406 pelanggan uji yang belanjanya < 100.
- **Residual:** Linear Regression membentuk garis miring di kiri (akibat tebakan negatif); kedua model makin melebar untuk belanja besar.
- **MAE Random Forest per segmen:** Hemat 80 (99% dari belanja rata-rata 81), Pemburu Diskon 217 (39%), Keluarga Mapan 242 (29%), Premium 300 (23%).

Angka bisa sedikit berbeda jika versi scikit-learn atau hasil pembersihan berbeda.

### File keluaran

`kartu_skor_model.csv`: tujuh baris pertanyaan direksi, ukuran yang menjawabnya, dan nilainya.

---

## Tentang materi pertemuan ini

**Business case.** Masih Juli 2014. Di rapat direksi, kamu berkata *"Model kita 87,7% akurat."* Manajer marketing bertanya berapa responden yang berhasil ditemukan; direktur keuangan bertanya berapa yang benar-benar merespons dari pelanggan yang dipilih. Akurasi tidak menjawab keduanya.

| Pertanyaan bisnis | Ukuran |
|---|---|
| Dari semua responden, berapa yang ditemukan? | Recall |
| Dari yang dipilih model, berapa yang benar? | Precision |
| Bagaimana menyeimbangkan keduanya? | F1 |
| Seberapa baik model mengurutkan pelanggan? | ROC-AUC, kurva precision-recall, kurva gain |
| Rata-rata perkiraan belanja meleset berapa? | MAE, RMSE |
| Seberapa jauh lebih baik daripada menebak rata-rata? | R² |

**Alur notebook.** Muat dua file → confusion matrix dengan TP/FP/FN/TN → precision, recall, F1 (manual dan scikit-learn) → membandingkan empat model dan `classification_report` → `class_weight="balanced"` → tabel dan kurva batas peluang → ROC, AUC, dan bukti pasangan → kurva precision-recall → kurva gain → regresi: MAE, RMSE, R², MAPE, residual, MAE per segmen → kartu skor model.

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Membaca confusion matrix dengan istilah TP, FP, FN, TN.
2. Menghitung dan menjelaskan precision, recall, dan F1, serta memilih yang sesuai biaya kesalahan.
3. Menjelaskan kenapa akurasi bisa turun padahal model menjadi lebih berguna.
4. Membaca kurva ROC dan AUC, kurva precision-recall, dan kurva gain.
5. Menilai model regresi dengan MAE, RMSE, R², dan MAPE, termasuk jebakan masing-masing.
6. Membaca residual dan meleset per kelompok.
7. Menyusun kartu skor model untuk orang non-teknis.

**Cara memakai notebook.** Buka `06_Model_Evaluation.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana, dan pastikan `prediksi_response_uji.csv` dan `marketing_segmen.csv` ada di folder `FOLDER` di Google Drive.

**Perkiraan waktu.** Slide ±40 menit, praktik notebook 70–80 menit.

**Output mahasiswa.** Notebook yang sudah dijalankan, `kartu_skor_model.csv` dengan satu baris tambahan, dan memo 4–6 kalimat untuk direksi.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
