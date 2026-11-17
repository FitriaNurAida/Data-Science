# Final Project — Data Science (Tengah Semester)

Tugas akhir tengah semester: selesaikan **satu masalah bisnis RetailKu** dengan machine learning, dari memahami masalah sampai memberi rekomendasi.

| File | Isi |
|---|---|
| `README FINAL PROJECT.md` | Ketentuan tugas, kamus data, dan contoh business case (dokumen ini) |
| `retailku_final_project.csv` | Dataset final project |

| Ketentuan | Keterangan |
|---|---|
| Bentuk kerja | [individu] |
| Batas pengumpulan | [Meet 9] |
| Presentasi | [Meet 9 & Meet 10] |
| Cara mengumpulkan | [https://drive.google.com/drive/folders/1ste7ew2ZPqxz1a98OTedb2aAU_mJ9eQ1?usp=sharing] |

---

## Dataset: `retailku_final_project.csv`

Dataset ini diambil dari dataset yang sudah kita pakai sejak Pertemuan 2 (Customer Personality Analysis), lalu disederhanakan menjadi **5 kolom prediktor dan 1 kolom respons**. Satu baris adalah satu pelanggan *RetailKu* (perusahaan fiktif, data asli).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Jumlah baris | 1.048 |
| Jumlah kolom | 6 |

### Kamus data

| Kolom | Peran bawaan | Tipe | Keterangan |
|---|---|---|---|
| `Income` | Prediktor | Angka | Pendapatan rumah tangga per tahun |
| `Children` | Prediktor | Angka (0–3) | Jumlah anak kecil dan remaja di rumah |
| `TotalSpend` | Prediktor | Angka | Total belanja selama 2 tahun di enam kategori produk (anggur, buah, daging, ikan, makanan manis, emas) |
| `NumDealsPurchases` | Prediktor | Angka | Jumlah pembelian yang memakai diskon |
| `Recency` | Prediktor | Angka (hari) | Jumlah hari sejak pembelian terakhir |
| `Response` | Respons | 0 / 1 | 1 jika pelanggan menerima tawaran pada kampanye terakhir |

> **Peran kolom bisa berubah sesuai business case.** Untuk supervised learning, kamu memilih satu kolom sebagai target. Untuk unsupervised learning, tidak ada target: `Response` **tidak boleh** dipakai untuk membentuk kelompok, tetapi boleh dipakai untuk **memvalidasi** kelompok yang terbentuk (seperti Pertemuan 4).

---

## Tugas

Pilih **satu** business case: **Contoh A**, **Contoh B**, atau **ide kamu sendiri**. Selesaikan dengan alur yang sudah kita pakai sejak Pertemuan 1 (CRISP-DM).

### Contoh A — Unsupervised learning: segmentasi untuk program loyalitas

> **Cerita.** RetailKu akan meluncurkan **program loyalitas** dengan beberapa tingkat keanggotaan. Manajer CRM bertanya: *"Pelanggan kita terdiri dari kelompok apa saja? Saya ingin setiap kelompok mendapat manfaat yang berbeda, bukan satu program untuk semua."*

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Pelanggan mana yang mirip satu sama lain? | Fitur apa yang dipakai untuk mengukur kemiripan? Perlukah scaling? |
| Ada berapa kelompok pelanggan? | Berapa K terbaik menurut elbow, silhouette, dan kebutuhan bisnis? |
| Seperti apa setiap kelompok? | Bagaimana profil tiap kelompok? Apa namanya? |
| Apakah kelompok ini berguna? | Apakah tingkat `Response` berbeda antar kelompok? |
| Manfaat apa untuk tiap kelompok? | Rekomendasi program loyalitas per kelompok |

**Langkah minimal:**

1. Tangani duplikat, lalu jelajahi sebaran tiap kolom.
2. Pilih fitur untuk clustering dan jelaskan alasannya. (Petunjuk: pikirkan apakah `Recency` menggambarkan *siapa* pelanggan itu atau *kapan* terakhir ia belanja.)
3. Lakukan scaling. Pertimbangkan transformasi log untuk kolom yang sangat miring.
4. Pilih K dengan metode elbow **dan** silhouette, lalu putuskan dengan pertimbangan bisnis.
5. Buat profil tiap kelompok (tabel dan/atau heatmap), beri nama yang mudah dipahami.
6. Validasi dengan `Response`: apakah ada kelompok yang lebih responsif?
7. Tulis rekomendasi manfaat program loyalitas untuk tiap kelompok.

### Contoh B — Supervised learning: memilih penerima kampanye berikutnya

> **Cerita.** Anggaran kampanye berikutnya hanya cukup untuk mengirim penawaran ke **30% pelanggan**. Manajer marketing bertanya: *"Siapa yang sebaiknya dikirimi? Saya ingin menjangkau sebanyak mungkin pelanggan yang benar-benar akan merespons."*

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Siapa yang kemungkinan besar merespons? | Klasifikasi dengan target `Response` |
| Apakah model kita lebih baik daripada tidak memakai model? | Baseline (`DummyClassifier`), confusion matrix |
| Model mana yang paling cocok? | Membandingkan minimal 2 model |
| Hanya 15% pelanggan yang merespons; apa dampaknya? | Kelas tidak seimbang: precision, recall, F1, `class_weight` |
| Jika dikirim ke 30% teratas, berapa responden yang terjangkau? | Mengurutkan dengan `predict_proba`, kurva gain |
| Faktor apa yang berhubungan dengan respons? | Koefisien atau feature importance |

**Langkah minimal:**

1. Tangani duplikat, lalu jelajahi hubungan tiap kolom dengan `Response`.
2. Bagi data latih dan uji (`stratify`, `random_state=42`). Pastikan persiapan data hanya di-*fit* pada data latih.
3. Buat baseline, lalu latih **minimal dua** model klasifikasi.
4. Nilai model dengan confusion matrix, precision, recall, F1, dan ROC-AUC, **bukan hanya akurasi**.
5. Coba tangani kelas tidak seimbang (misalnya `class_weight="balanced"`) dan bandingkan hasilnya.
6. Hitung berapa persen responden yang terjangkau jika penawaran dikirim ke 30% pelanggan dengan peluang tertinggi.
7. Tulis rekomendasi: model mana yang dipakai, siapa yang dikirimi, dan apa keterbatasannya.

### Ide sendiri

Kamu boleh membuat business case sendiri dari keenam kolom ini. Beberapa arah yang mungkin:

- **Regresi:** memperkirakan `TotalSpend` dari kolom lain.
- **Klasifikasi lain:** misalnya memprediksi pelanggan "pemburu diskon" (target dibuat dari `NumDealsPurchases`).
- **Gabungan:** membuat segmen (unsupervised), lalu membandingkan model klasifikasi per segmen.

Syarat ide sendiri:

- Pertanyaan bisnisnya jelas: siapa yang bertanya, dan keputusan apa yang akan diambil dari hasilmu.
- Jika memakai supervised learning, **target tidak boleh bocor**: kolom yang dipakai membuat target tidak boleh ikut menjadi prediktor.
- Konsultasikan idemu ke dosen paling lambat [tanggal].

---

## Yang dikumpulkan

1. **Notebook** (`.ipynb`) yang bisa dijalankan dari atas sampai bawah tanpa error, dengan struktur:

   | Bagian | Isi |
   |---|---|
   | 1. Business understanding | Cerita, pertanyaan bisnis, dan pertanyaan data |
   | 2. Data understanding | Eksplorasi: ukuran, tipe, sebaran, hubungan antar kolom |
   | 3. Data preparation | Duplikat (wajib) dan langkah lain beserta alasannya |
   | 4. Modeling | Model yang dipakai dan alasannya |
   | 5. Evaluation | Ukuran yang sesuai business case, beserta artinya |
   | 6. Rekomendasi | Jawaban untuk pertanyaan bisnis, dalam bahasa non-teknis |

   Setiap grafik dan tabel diikuti 1–2 kalimat penjelasan: *apa yang terlihat* dan *apa artinya bagi RetailKu*.

2. **Slide presentasi** ([… slide]) untuk manajer RetailKu: masalah, pendekatan, hasil utama, rekomendasi, keterbatasan.

## Penilaian

| Komponen | Bobot | Yang dinilai |
|---|---|---|
| Business understanding | 15% | Pertanyaan bisnis jelas dan diterjemahkan menjadi pertanyaan data |
| Data understanding | 15% | Eksplorasi relevan, setiap grafik dibaca dan dimaknai |
| Data preparation | 10% | Duplikat ditangani dan dilaporkan; langkah lain beralasan; tidak ada kebocoran data |
| Modeling | 20% | Metode sesuai masalah; parameter penting (K, model, `random_state`) dijelaskan |
| Evaluation | 20% | Ukuran sesuai business case; tidak hanya akurasi; hasil dibaca dengan jujur |
| Rekomendasi bisnis | 15% | Menjawab pertanyaan bisnis; bisa ditindaklanjuti; menyebut keterbatasan |
| Komunikasi | 5% | Notebook rapi dan berjalan; slide jelas untuk orang non-teknis |

## Pengingat dari pertemuan sebelumnya

- **Pertemuan 2–3:** duplikat bisa menyebabkan kebocoran data. Fit scaler dan imputer **hanya di data latih**.
- **Pertemuan 4:** clustering bergantung pada jarak, jadi **scaling wajib**. Pilih K dengan angka **dan** pertimbangan bisnis.
- **Pertemuan 5:** selalu bandingkan dengan **baseline**. Waspadai overfitting (skor latih jauh di atas skor uji).
- **Pertemuan 6:** pada kelas yang tidak seimbang, **akurasi bisa menipu**. Pilih precision atau recall sesuai biaya kesalahan.

---

## Lisensi dan sitasi

Diolah dari Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
