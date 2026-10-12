# Pertemuan 2 — Preprocessing Data 1

Materi pertemuan kedua mata kuliah Data Science: data cleaning dan kualitas data.

| File | Isi |
|---|---|
| `README.md` | Metadata dataset (dokumen ini) |
| `02_Preprocessing_Data_1.pptx` | Slide materi |
| `02_Preprocessing_Data_1.ipynb` | Notebook praktik untuk Google Colab |

Struktur repositori yang diasumsikan notebook:

```
data/
└── marketing_campaign.csv        <- dipakai Pertemuan 2 sampai 7
pertemuan-02/
├── README.md
├── 02_Preprocessing_Data_1.pptx
└── 02_Preprocessing_Data_1.ipynb
```

---

## Dataset: Customer Personality Analysis

Data pelanggan sebuah perusahaan ritel yang menjual produk kebutuhan rumah tangga (anggur, buah, daging, ikan, makanan manis, dan produk emas) lewat toko, situs web, dan katalog. Satu baris adalah satu pelanggan, berisi profilnya, belanjanya selama dua tahun terakhir, respons terhadap kampanye promosi, dan kanal belanja yang dipakai.

Dalam business case di kelas, perusahaan ini kita sebut **RetailKu** (nama fiktif). Datanya asli.

**Dataset ini dipakai dari Pertemuan 2 sampai Pertemuan 7**, supaya mahasiswa cukup sekali mengenal kolom-kolomnya dan bisa fokus pada tekniknya.

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Customer Personality Analysis |
| Sumber | Kaggle |
| Halaman dataset | https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| Pengunggah | Akash Patel (`imakash3011`) |
| Lisensi | CC0: Public Domain (menurut proyek-proyek lain yang memakai dataset ini; cek kembali di halaman Kaggle) |
| Nama file | `marketing_campaign.csv` |
| Pemisah kolom | **Tab** (`\t`), bukan koma |
| Jumlah baris | 2.240 |
| Jumlah kolom | 29 |
| Rentang tanggal pendaftaran pelanggan | 30 Juli 2012 – 29 Juni 2014 |
| Ada nilai kosong | Ya, pada kolom `Income` |
| Mata uang | Tidak disebutkan di sumber data |

### Kamus data

Kolom-kolomnya bisa dikelompokkan menjadi empat.

**Profil pelanggan**

| Kolom | Tipe | Keterangan |
|---|---|---|
| `ID` | Label | Nomor unik pelanggan |
| `Year_Birth` | Angka | Tahun lahir pelanggan |
| `Education` | Kategori | Jenjang pendidikan: `Basic`, `Graduation`, `2n Cycle`, `Master`, `PhD` |
| `Marital_Status` | Kategori | Status pernikahan |
| `Income` | Angka | Pendapatan rumah tangga per tahun |
| `Kidhome` | Angka | Jumlah anak kecil di rumah |
| `Teenhome` | Angka | Jumlah remaja di rumah |
| `Dt_Customer` | Tanggal (tersimpan sebagai teks, format hari-bulan-tahun) | Tanggal pelanggan mendaftar |
| `Recency` | Angka | Jumlah hari sejak pembelian terakhir |
| `Complain` | 0 / 1 | 1 jika pelanggan pernah mengajukan keluhan dalam 2 tahun terakhir |

**Produk** (jumlah belanja selama 2 tahun terakhir)

| Kolom | Keterangan |
|---|---|
| `MntWines` | Belanja anggur (wine) |
| `MntFruits` | Belanja buah |
| `MntMeatProducts` | Belanja daging |
| `MntFishProducts` | Belanja ikan |
| `MntSweetProducts` | Belanja makanan manis |
| `MntGoldProds` | Belanja produk emas |

**Promosi**

| Kolom | Tipe | Keterangan |
|---|---|---|
| `NumDealsPurchases` | Angka | Jumlah pembelian dengan diskon |
| `AcceptedCmp1` … `AcceptedCmp5` | 0 / 1 | 1 jika pelanggan menerima tawaran pada kampanye ke-1 sampai ke-5 |
| `Response` | 0 / 1 | 1 jika pelanggan menerima tawaran pada **kampanye terakhir** |

**Kanal belanja**

| Kolom | Keterangan |
|---|---|
| `NumWebPurchases` | Jumlah pembelian lewat situs web |
| `NumCatalogPurchases` | Jumlah pembelian lewat katalog |
| `NumStorePurchases` | Jumlah pembelian langsung di toko |
| `NumWebVisitsMonth` | Jumlah kunjungan ke situs web dalam sebulan terakhir |

**Kolom lain**

| Kolom | Keterangan |
|---|---|
| `Z_CostContact` | Isinya sama di semua baris (3). Tidak dijelaskan di dokumentasi dataset |
| `Z_Revenue` | Isinya sama di semua baris (11). Tidak dijelaskan di dokumentasi dataset |

### Hal yang perlu diwaspadai

Dataset ini kotor **secara alami**, sehingga cocok untuk latihan data cleaning. Semua hal berikut ditemukan mahasiswa sendiri di notebook Pertemuan 2.

| Temuan | Dimensi kualitas |
|---|---|
| File `.csv` yang ternyata dipisahkan tab | Format |
| `Dt_Customer` tersimpan sebagai teks | Valid |
| Dua kolom yang isinya sama di semua baris | — |
| Pelanggan yang tercatat lebih dari sekali dengan `ID` berbeda (duplikat yang tidak terdeteksi oleh `df.duplicated()` biasa) | Unik |
| Label status pernikahan yang tidak valid atau bersinonim | Konsisten, Valid |
| Dua label pendidikan untuk jenjang yang setara | Konsisten |
| Tahun lahir yang tidak masuk akal | Valid |
| Nilai pendapatan yang janggal | Valid |
| Pendapatan kosong | Lengkap |
| Banyak outlier di kolom belanja, yang ternyata nilai nyata | — |

Jumlah persis tiap temuan sengaja tidak dituliskan di sini. Notebook menghitungnya langsung dari data.

### Cara mendapatkan data

1. Unduh dari halaman Kaggle di atas (perlu login).
2. Simpan `marketing_campaign.csv` di folder `data/` pada repositori kelas.
3. Ganti `[username]` dan `[nama-repo]` pada `URL_DATA` di bagian 3 notebook. Notebook lalu mengunduh file langsung dari repositori, jadi mahasiswa tidak perlu akun Kaggle.

Jika unduhan gagal, mahasiswa bisa mengunggah file secara manual lewat ikon folder di sisi kiri Colab.

### Membaca data dengan pandas

```python
import pandas as pd

df = pd.read_csv("marketing_campaign.csv", sep="\t")
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
```

### Pemakaian dalam mata kuliah

| Pertemuan | Topik | Dataset ini dipakai untuk |
|---|---|---|
| 2 | Preprocessing Data 1 | Data cleaning: duplikat, kategori tidak konsisten, nilai tidak valid, outlier, nilai kosong |
| 3 | Preprocessing Data 2 | Feature engineering: umur, jumlah anak, total belanja, lama menjadi pelanggan; encoding dan scaling |
| 4 | Unsupervised Learning | Segmentasi pelanggan |
| 5 | Classification & Regression | Klasifikasi: memprediksi `Response`. Regresi: memprediksi total belanja |
| 6 | Model Evaluation | Metrik klasifikasi (respons hanya sebagian kecil pelanggan) dan metrik regresi |
| 7 | Model Evaluation 2 | Cross-validation, tuning, threshold, dan simulasi biaya-manfaat kampanye |

---

## Tentang materi pertemuan ini

**Business case.** Dalam cerita, sekarang Juli 2014. Mahasiswa sudah dua setengah tahun bekerja di RetailKu dan baru pindah ke tim marketing. Seorang anak magang membuat laporan profil pelanggan dari data mentah: *"2.240 pelanggan, berumur 18 sampai 121 tahun, dengan 8 jenis status pernikahan."* Manajer marketing curiga dan meminta datanya dicek sebelum dibawa ke direksi.

| Pertanyaan bisnis | Pertanyaan data | Dimensi |
|---|---|---|
| Berapa jumlah pelanggan kita sebenarnya? | Adakah pelanggan yang tercatat lebih dari sekali? | Unik |
| Benarkah ada pelanggan berumur 121 tahun? | Nilai mana yang tidak masuk akal? | Valid |
| Bisakah angka pendapatan dipercaya? | Adakah nilai kosong atau nilai ekstrem? | Lengkap |
| Kenapa status pernikahan ada delapan? | Adakah label yang tidak konsisten atau tidak valid? | Konsisten |

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Mengaudit data dari empat dimensi kualitas: lengkap, valid, konsisten, dan unik.
2. Memperbaiki tipe data, termasuk mengubah teks menjadi tanggal dengan format eksplisit.
3. Menemukan duplikat, termasuk duplikat dengan ID berbeda.
4. Menyeragamkan kategori yang tidak konsisten.
5. Mendeteksi outlier dengan IQR dan z-score, lalu membedakan kesalahan dengan nilai nyata.
6. Menangani nilai kosong dan menjelaskan pilihan antara mean dan median.
7. Mendokumentasikan keputusan pembersihan dalam log.

**Cara memakai notebook.** Buka `02_Preprocessing_Data_1.ipynb` di GitHub, lalu klik lencana **Open in Colab**. Sebelumnya, ganti `[username]` dan `[nama-repo]` pada lencana dan pada `URL_DATA`.

**Output mahasiswa.** Notebook yang sudah dijalankan, file `marketing_campaign_bersih.csv`, log pembersihan, dan memo 4–6 kalimat untuk manajer marketing.

---

## Lisensi dan sitasi

Customer Personality Analysis, diunggah oleh Akash Patel di Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
