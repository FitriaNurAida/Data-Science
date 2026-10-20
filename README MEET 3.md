# Pertemuan 3 — Preprocessing Data 2

Materi pertemuan ketiga mata kuliah Data Science: feature engineering dan menyiapkan data untuk model.

| File | Isi |
|---|---|
| `README.md` | Metadata dataset dan kamus fitur (dokumen ini) |
| `03_Preprocessing_Data_2.pptx` | Slide materi |
| `03_Preprocessing_Data_2.ipynb` | Notebook praktik untuk Google Colab |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-03/
├── README.md
├── 03_Preprocessing_Data_2.pptx
└── 03_Preprocessing_Data_2.ipynb
```

---

## Dataset: Customer Personality Analysis

Dataset yang sama dengan Pertemuan 2: data pelanggan sebuah perusahaan ritel yang menjual kebutuhan rumah tangga lewat toko, situs web, dan katalog. Satu baris adalah satu pelanggan. Dalam business case, perusahaan ini disebut **RetailKu** (nama fiktif).

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle, diunggah oleh Akash Patel (`imakash3011`) |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| Nama file | `marketing_campaign.csv` |
| Pemisah kolom | Tab (`\t`) |
| Data mentah | 2.240 baris × 29 kolom |
| Setelah pembersihan Pertemuan 2 | 2.051 baris × 27 kolom |
| Target untuk Pertemuan 5–7 | `Response`: 1 jika pelanggan menerima tawaran pada kampanye terakhir |

Penjelasan lengkap setiap kolom asli ada di `README.md` Pertemuan 2.

### Membaca dan membersihkan data

```python
import pandas as pd

mentah = pd.read_csv("marketing_campaign.csv", sep="\t")
bersih = bersihkan_data(mentah)   # fungsi di bagian 2 notebook, merangkum keputusan Pertemuan 2
```

Perbedaan dengan Pertemuan 2: nilai kosong `Income` **tidak** diisi di fungsi `bersihkan_data()`, melainkan di dalam pipeline, agar median hanya dihitung dari data latih.

---

## Kamus fitur

Fitur baru yang dibuat di notebook (fungsi `buat_fitur()`). Tanggal acuan cerita adalah **1 Juli 2014**.

| Fitur | Cara menghitung | Arti bagi bisnis |
|---|---|---|
| `Age` | 2014 − `Year_Birth` | Umur pelanggan |
| `Tenure_days` | 1 Juli 2014 − `Dt_Customer`, dalam hari | Lama menjadi pelanggan |
| `Children` | `Kidhome` + `Teenhome` | Jumlah anak di rumah |
| `TotalSpend` | Jumlah 6 kolom `Mnt...` | Nilai belanja selama 2 tahun terakhir |
| `TotalPurchases` | `NumWebPurchases` + `NumCatalogPurchases` + `NumStorePurchases` | Jumlah transaksi di semua kanal |
| `TotalAcceptedPrev` | Jumlah `AcceptedCmp1` sampai `AcceptedCmp5` | Berapa kampanye sebelumnya yang diterima. **Tidak** termasuk `Response` |
| `AvgSpendPerPurchase` | `TotalSpend` ÷ `TotalPurchases` (kosong jika `TotalPurchases` = 0) | Belanja rata-rata per transaksi |
| `AgeGroup` | `Age` dikelompokkan: ≤30, 31–45, 46–60, >60 | Kelompok umur untuk laporan |
| `TotalSpend_log` | log(1 + `TotalSpend`) | Total belanja dengan sebaran yang lebih seimbang, untuk model |
| `Education_ord` | Basic = 0, Graduation = 1, Master = 2, PhD = 3 | Jenjang pendidikan sebagai angka berurutan |

### Fitur yang dipakai pipeline

| Jalur | Kolom | Langkah |
|---|---|---|
| Angka | `Income`, `Age`, `Tenure_days`, `Recency`, `Children`, `TotalSpend_log`, `TotalPurchases`, `AvgSpendPerPurchase`, `NumDealsPurchases`, `NumWebVisitsMonth`, `TotalAcceptedPrev`, `Complain` | Isi kosong dengan median, lalu `StandardScaler` |
| Ordinal | `Education` | `OrdinalEncoder` dengan urutan Basic < Graduation < Master < PhD |
| Nominal | `Marital_Status` | `OneHotEncoder` (kategori baru diabaikan) |

Hasilnya 18 kolom siap model. Data dibagi 80% latih dan 20% uji dengan `stratify` pada `Response` dan `random_state=42`.

### File keluaran

`marketing_fitur.csv`: data bersih beserta semua fitur baru, **belum** di-scaling. File ini menjadi titik awal Pertemuan 4 dan 5.

---

## Tentang materi pertemuan ini

**Business case.** Masih Juli 2014. Direksi menyetujui kampanye ke-7 dengan syarat lebih tepat sasaran. Data scientist senior meminta *"tabel siap model: semua kolom berupa angka, skalanya setara, tanpa informasi yang bocor, dan prosesnya bisa diulang untuk pelanggan baru."*

| Pertanyaan bisnis | Pertanyaan data / teknik |
|---|---|
| Apa saja yang menggambarkan seorang pelanggan? | Fitur baru apa yang bisa dibuat? (feature engineering) |
| Bagaimana model membaca "PhD" atau "Married"? | Bagaimana mengubah kategori menjadi angka? (encoding) |
| Apakah pendapatan dan jumlah anak bisa dibandingkan begitu saja? | Bagaimana menyetarakan skala? (scaling) |
| Bagaimana memastikan model diuji dengan adil? | Bagaimana membagi data dan mencegah kebocoran? (split, leakage) |
| Bagaimana jika besok ada pelanggan baru? | Bagaimana membuat semua langkah bisa diulang? (pipeline) |

**Alur notebook.** Mahasiswa lebih dulu menjelajahi data bersih dari sudut pandang model dan mencatat apa saja yang belum siap. Setelah itu baru masuk ke feature engineering, encoding, scaling, split dan kebocoran data, lalu pipeline.

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Menilai apakah data yang sudah bersih siap dipakai model.
2. Membuat fitur baru yang bermakna bagi bisnis: dari tanggal, agregasi, rasio, binning, dan transformasi log.
3. Memilih antara ordinal encoding dan one-hot encoding, serta menjelaskan risiko label encoding.
4. Menjelaskan kenapa scaling diperlukan dan membedakan standardization dengan min-max scaling.
5. Membagi data latih dan data uji dengan stratifikasi.
6. Mengenali tiga bentuk kebocoran data.
7. Menyusun `Pipeline` dan `ColumnTransformer` yang di-fit hanya pada data latih.

**Cara memakai notebook.** Buka `03_Preprocessing_Data_2.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana dan pada `URL_DATA`.

**Output mahasiswa.** Notebook yang sudah dijalankan, file `marketing_fitur.csv`, kamus fitur untuk minimal 6 fitur baru, dan 3–4 kalimat tentang cara mencegah kebocoran data.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
