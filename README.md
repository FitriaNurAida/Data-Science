# Pertemuan 1 — Introduction Data Science

Materi pertemuan pertama mata kuliah Data Science: pengenalan data science dan EDA pertama dengan data transaksi retail.

| File | Isi |
|---|---|
| `README.md` | Metadata dataset (dokumen ini) |
| `01_Introduction_Data_Science.pptx` | Slide materi |
| `01_Introduction_Data_Science.ipynb` | Notebook praktik untuk Google Colab |

---

## Dataset: Online Retail II

Seluruh transaksi sebuah toko online di Inggris selama dua tahun, antara **1 Desember 2009** dan **9 Desember 2011**. Toko ini tidak memiliki gerai fisik dan terutama menjual pernak-pernik hadiah untuk segala acara (*all-occasion gift-ware*). Banyak pelanggannya adalah pedagang grosir.

Dalam business case di kelas, toko ini kita sebut **RetailKu** (nama fiktif). Datanya asli.

### Informasi umum

| Atribut | Nilai |
|---|---|
| Nama dataset | Online Retail II |
| Sumber | UCI Machine Learning Repository |
| Halaman dataset | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| Pembuat | Daqing Chen |
| Tanggal donasi ke UCI | 20 September 2019 |
| DOI | [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D) |
| Lisensi | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Bidang | Bisnis |
| Karakteristik | Multivariate, Sequential, Time-Series, Text |
| Tugas yang umum | Classification, Regression, Clustering |
| Jumlah baris | 1.067.371 |
| Jumlah kolom | 8 |
| Ada nilai kosong | Ya |
| Periode data | 1 Desember 2009 – 9 Desember 2011 |
| Mata uang | Poundsterling (£) |
| Nama file | `online_retail_II.xlsx` |
| Ukuran file | 43,5 MB |

### Struktur file

File Excel terdiri dari **dua sheet** dengan kolom yang sama. Notebook membaca keduanya lalu menggabungkannya.

| Sheet | Periode | Jumlah baris |
|---|---|---|
| `Year 2009-2010` | 1 Des 2009 – 9 Des 2010 | 525.461 |
| `Year 2010-2011` | 1 Des 2010 – 9 Des 2011 | 541.910 |
| **Total** | | **1.067.371** |

> Rentang tanggal kedua sheet beririsan pada 1–9 Desember 2010. Notebook menampilkan jumlah baris dan rentang tanggal tiap sheet saat dijalankan, sehingga angka di tabel ini bisa dicek langsung.

### Kamus data

**Satu baris = satu jenis barang dalam satu invoice (nota).** Satu invoice bisa terdiri dari banyak baris.

| Kolom di file | Nama di halaman UCI | Nama di notebook | Tipe | Keterangan |
|---|---|---|---|---|
| `Invoice` | `InvoiceNo` | `Invoice` | Nominal | Nomor invoice, 6 digit, unik untuk tiap transaksi. Jika diawali huruf **C**, transaksi tersebut adalah pembatalan |
| `StockCode` | `StockCode` | `StockCode` | Nominal | Kode produk, 5 digit, unik untuk tiap produk |
| `Description` | `Description` | `Description` | Nominal | Nama produk |
| `Quantity` | `Quantity` | `Quantity` | Numerik | Jumlah tiap produk per transaksi |
| `InvoiceDate` | `InvoiceDate` | `InvoiceDate` | Tanggal & waktu | Tanggal dan jam transaksi dibuat |
| `Price` | `UnitPrice` | `Price` | Numerik | Harga per unit produk dalam poundsterling (£) |
| `Customer ID` | `CustomerID` | `CustomerID` | Nominal | Nomor pelanggan, 5 digit, unik untuk tiap pelanggan |
| `Country` | `Country` | `Country` | Nominal | Negara tempat tinggal pelanggan |

Nama kolom di dalam file sedikit berbeda dari nama di halaman UCI. Notebook menyamakan `Customer ID` menjadi `CustomerID` supaya mudah diketik.

Kolom turunan yang dibuat di notebook:

| Kolom | Rumus | Keterangan |
|---|---|---|
| `Sheet` | nama sheet asal | Penanda asal baris, dipakai untuk memeriksa irisan tanggal |
| `Revenue` | `Quantity × Price` | Nilai penjualan per baris (£) |
| `Bulan` | dari `InvoiceDate` | Periode bulanan, misalnya `2010-03` |
| `Kuartal` | dari `InvoiceDate` | Periode kuartalan, misalnya `2011Q1` |

### Hal yang perlu diwaspadai

Dataset ini sengaja dipilih karena **tidak bersih**, sama seperti data di dunia kerja. Sebagian besar hal berikut akan ditemukan mahasiswa sendiri di notebook, dan dibereskan di Pertemuan 2.

| Temuan | Penjelasan | Dibahas di |
|---|---|---|
| Invoice berawalan `C` | Transaksi yang dibatalkan, tercatat dengan `Quantity` negatif | P1 (dikenali), P2 |
| `Customer ID` kosong | Sebagian transaksi tidak memiliki nomor pelanggan | P1 (dihitung), P2 |
| `Description` kosong | Sebagian kecil baris tidak memiliki nama produk | P2 |
| `Quantity` negatif | Pembatalan, retur, atau penyesuaian stok | P1 (dikenali), P2 |
| `Price` nol atau negatif | Bukan penjualan biasa, misalnya penyesuaian | P2 |
| `StockCode` bukan produk | Sebagian kode mewakili hal selain barang, misalnya ongkos kirim atau penyesuaian manual | P2 |
| Baris duplikat | Termasuk irisan 1–9 Desember 2010 di kedua sheet | P1 (dibuang), P2 |
| Periode tidak lengkap | Desember 2011 hanya sampai tanggal 9, sehingga bulan dan kuartal terakhir tidak bisa dibandingkan langsung | P1, P9 |
| Satu negara dominan | Sebagian besar transaksi berasal dari satu negara | P1 |

Jumlah persis untuk tiap temuan sengaja tidak dituliskan di sini. Notebook menghitungnya langsung dari data.

### Cara mendapatkan data

**Otomatis.** Notebook mengunduh sendiri file dari UCI saat cell pengambilan data dijalankan:

```python
import urllib.request, zipfile

URL_DATA = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
urllib.request.urlretrieve(URL_DATA, "online_retail_ii.zip")
with zipfile.ZipFile("online_retail_ii.zip") as z:
    z.extractall(".")          # menghasilkan online_retail_II.xlsx
```

**Manual.** Jika unduhan otomatis gagal:

1. Buka https://archive.ics.uci.edu/dataset/502/online+retail+ii lalu klik **Download**.
2. Buka file zip untuk mendapatkan `online_retail_II.xlsx`.
3. Unggah ke Colab (ikon folder di sisi kiri) atau simpan di Google Drive, lalu ikuti petunjuk di bagian 3 notebook.

File data **tidak disertakan** di repositori ini agar repositori tetap ringan.

### Membaca data dengan pandas

```python
import pandas as pd

sheets = pd.read_excel("online_retail_II.xlsx", sheet_name=None)   # baca semua sheet
df = pd.concat(sheets.values(), ignore_index=True)
df = df.rename(columns={"Customer ID": "CustomerID"})
```

Membaca file ini memakan waktu beberapa menit karena ukurannya besar.

### Pemakaian dalam mata kuliah

| Pertemuan | Topik | Dataset ini dipakai untuk |
|---|---|---|
| 1 | Introduction Data Science | EDA pertama, tren penjualan bulanan dan kuartalan |
| 2 | Preprocessing Data 1 | Data cleaning: nilai kosong, pembatalan, duplikat, outlier |
| 3 | Preprocessing Data 2 | Feature engineering: fitur RFM per pelanggan |
| 4 | Unsupervised Learning | Segmentasi pelanggan |
| 9 | Forecast with ARIMA | Peramalan penjualan |

---

## Tentang materi pertemuan ini

**Business case.** Mahasiswa berperan sebagai data analyst baru di RetailKu. Dalam cerita, sekarang awal April 2011 dan manajer bertanya: *"Penjualan kuartal pertama tahun ini turun dibanding kuartal sebelumnya. Kenapa? Apa kita perlu khawatir?"*

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Menjelaskan apa itu data science dan contoh penerapannya.
2. Membedakan peran Data Analyst, Data Scientist, Data Engineer, dan ML Engineer.
3. Menyebutkan enam tahap CRISP-DM.
4. Membedakan analitik descriptive, diagnostic, predictive, dan prescriptive.
5. Memuat data ke pandas dan mengenalinya dengan `head()`, `info()`, dan `describe()`.
6. Membuat grafik tren penjualan dan mengubah pertanyaan bisnis menjadi pertanyaan data.

**Cara memakai notebook.** Buka `01_Introduction_Data_Science.ipynb` di GitHub, lalu klik lencana **Open in Colab** di bagian atas. Sebelum itu, ganti `[username]` dan `[nama-repo]` pada tautan lencana dengan alamat repositori ini.

**Output mahasiswa.** Notebook EDA yang sudah dijalankan, jawaban 3–4 kalimat untuk manajer, dan 3 pertanyaan bisnis lain yang bisa dijawab dengan data ini.

---

## Lisensi dan sitasi

Dataset dilisensikan di bawah [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): boleh dibagikan dan diadaptasi untuk tujuan apa pun selama mencantumkan sumbernya.

Sitasi sesuai halaman UCI:

> Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
