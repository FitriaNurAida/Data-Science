# Pertemuan 4 — Unsupervised Learning

Materi pertemuan keempat mata kuliah Data Science: unsupervised learning dan segmentasi pelanggan dengan K-Means.

| File | Isi |
|---|---|
| `README MEET 4.md` | Metadata dataset, fitur segmentasi, dan profil segmen (dokumen ini) |
| `04_Unsupervised_Learning.pptx` | Slide materi (23 slide, dengan catatan pembicara) |
| `04_Unsupervised_Learning.ipynb` | Notebook praktik untuk Google Colab |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-04/
├── README.md
├── 04_Unsupervised_Learning.pptx
└── 04_Unsupervised_Learning.ipynb
```

---

## Dataset: Customer Personality Analysis

Dataset yang sama dengan Pertemuan 2 dan 3: data pelanggan sebuah perusahaan ritel yang menjual kebutuhan rumah tangga lewat toko, situs web, dan katalog. Satu baris adalah satu pelanggan. Dalam business case, perusahaan ini disebut **RetailKu** (nama fiktif).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle, diunggah oleh Akash Patel (`imakash3011`) |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| File masukan | `marketing_fitur.csv`, hasil Pertemuan 3 (data bersih + fitur baru, belum di-scaling) |
| Jumlah baris | 2.026 pelanggan (hasil pembersihan di notebook Pertemuan 2, termasuk membuang baris dengan `Income` kosong) |
| Target | **Tidak ada.** `Response` hanya dipakai untuk validasi setelah segmen terbentuk |

Penjelasan kolom asli ada di `README MEET 2.md`; penjelasan fitur baru ada di `README MEET 3.md`.

### Membaca data

```python
import pandas as pd

df = pd.read_csv("marketing_fitur.csv")   # file keluaran Pertemuan 3
```

---

## Fitur untuk segmentasi

| Fitur | Arti bagi bisnis |
|---|---|
| `Income` | Daya beli |
| `TotalSpend_log` | Nilai pelanggan (log agar tidak didominasi segelintir pelanggan besar) |
| `TotalPurchases` | Seberapa sering berbelanja |
| `Children` | Tahap hidup / keluarga |
| `NumDealsPurchases` | Kepekaan terhadap diskon |
| `NumWebVisitsMonth` | Keterlibatan di kanal online |

Sengaja **tidak** dipakai: `ID` (hanya nomor), `Response` (untuk validasi), dan kolom kategori seperti `Marital_Status` (jarak antarkategori sulit dimaknai).

Langkah persiapan: `SimpleImputer(strategy="median")` lalu `StandardScaler()`, dalam satu `Pipeline`. Tidak ada train-test split karena tidak ada target yang ditebak.

### Model

`KMeans(n_clusters=4, n_init=10, random_state=42)`. Nomor segmen diurutkan dari median `TotalSpend` terendah (0) ke tertinggi (3) agar sama di semua komputer.

| K | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|
| Inertia | 6.997 | 5.213 | **4.496** | 3.896 | 3.585 |
| Silhouette | 0,391 | 0,367 | **0,342** | 0,304 | 0,295 |

K = 4 dipilih dari gabungan siku elbow (sekitar 3–4), silhouette yang masih baik, dan kebutuhan bisnis (empat penawaran masih bisa dikelola tim marketing).

### Profil segmen (kunci jawaban untuk dosen)

| Segmen | Nama usulan | Pelanggan | Pendapatan median | Belanja median (2 th) | Ciri menonjol | % merespons |
|---|---|---|---|---|---|---|
| 0 | Hemat | 825 (41%) | 33.200 | 57 | Belanja kecil, sering mengunjungi web tapi jarang membeli | 9,9% |
| 1 | Pemburu Diskon | 205 (10%) | 50.200 | 467 | 6,6 pembelian diskon, anak paling banyak | 18,5% |
| 2 | Keluarga Mapan | 536 (26%) | 60.500 | 765 | Belanja rutin dan cukup besar di toko dan web | 11,4% |
| 3 | Premium | 460 (23%) | 77.400 | 1.335 | Hampir tanpa anak, suka katalog, jarang pakai diskon | 28,0% |

Rata-rata respons semua pelanggan: 15,3%. Angka bisa sedikit berbeda jika versi scikit-learn atau hasil pembersihan berbeda.

### File keluaran

`marketing_segmen.csv`: isi `marketing_fitur.csv` ditambah kolom `Segmen` (0–3) dan `NamaSegmen`. Bisa dipakai Pertemuan 5 sebagai fitur tambahan atau untuk menganalisis hasil prediksi per segmen.

---

## Tentang materi pertemuan ini

**Business case.** Masih Juli 2014. Manajer marketing tidak mau lagi mengirim satu penawaran yang sama ke semua pelanggan (hanya sekitar 15% yang merespons). Ia meminta: *"Kelompokkan pelanggan kita menjadi beberapa segmen yang jelas bedanya, beri nama, dan sarankan penawaran untuk tiap segmen."* Masalahnya, tidak ada kolom "segmen" di data.

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Pelanggan mana yang mirip satu sama lain? | Bagaimana mengukur kemiripan? (jarak + scaling) |
| Bagaimana mengelompokkan ribuan pelanggan? | Algoritma apa yang dipakai? (K-Means) |
| Sebaiknya ada berapa segmen? | Bagaimana memilih K? (elbow, silhouette, bisnis) |
| Seperti apa tiap segmen? | Bagaimana membuat profil dan nama segmen? |
| Pelanggan baru masuk segmen mana? | Bagaimana memakai ulang model? (pipeline) |

**Alur notebook.** Eksplorasi visual → mengulang konsep jarak dan scaling → K-Means manual langkah demi langkah → memilih fitur dan eksperimen tanpa scaling → memilih K → profil, heatmap, validasi dengan `Response`, dan penamaan segmen → PCA untuk visualisasi → sekilas hierarchical clustering (dendrogram) → menetapkan segmen pelanggan baru dengan pipeline.

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Membedakan supervised dan unsupervised learning, serta menyebutkan contoh tugas unsupervised.
2. Menjelaskan cara kerja K-Means: centroid, pengelompokan, pembaruan, dan iterasi.
3. Menjelaskan kenapa scaling wajib sebelum clustering.
4. Memilih jumlah segmen dengan metode elbow, silhouette score, dan pertimbangan bisnis.
5. Membuat profil segmen, membaca heatmap z-score, dan memberi nama segmen.
6. Memakai PCA untuk memvisualisasikan segmen dan mengenal metode clustering lain (hierarchical, DBSCAN).
7. Menetapkan segmen untuk pelanggan baru dengan pipeline.

**Cara memakai notebook.** Buka `04_Unsupervised_Learning.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana, dan sesuaikan `FILE_DATA` dengan lokasi `marketing_fitur.csv` di Google Drive.

**Perkiraan waktu.** Slide ±35 menit, praktik notebook 60–70 menit.

**Output mahasiswa.** Notebook yang sudah dijalankan, file `marketing_segmen.csv`, kartu persona untuk 4 segmen, dan memo 4–6 kalimat untuk manajer marketing tentang segmen prioritas dan penawaran kampanye ke-7.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
