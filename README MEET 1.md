# Pertemuan 1 — Introduction Data Science

Materi pertemuan pertama mata kuliah Data Science: pengenalan data science dan EDA pertama dengan data transaksi retail.

| File | Isi |
|---|---|
| `README.md` | Metadata dataset (dokumen ini) |
| `01_Introduction_Data_Science.pdf` | Slide materi |
| `01_Introduction_Data_Science.ipynb` | Notebook praktik untuk Google Colab |

---

## Dataset: Online Retail II

Seluruh transaksi sebuah toko online di Inggris selama dua tahun, antara **Desember 2010** dan **Desember 2011**. Toko ini tidak memiliki gerai fisik dan terutama menjual pernak-pernik hadiah untuk segala acara (*all-occasion gift-ware*). Banyak pelanggannya adalah pedagang grosir.

Dalam business case di kelas, toko ini kita sebut **RetailKu** (nama fiktif). Datanya asli.

Download Dataset : https://drive.google.com/file/d/1loJRAAZCawSGkrWuhxecixQtKqC8IZZm/view?usp=sharing 

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
| Mata uang | Poundsterling (£) |

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
| `CustomerID` | `CustomerID` | `CustomerID` | Nominal | Nomor pelanggan, 5 digit, unik untuk tiap pelanggan |
| `Country` | `Country` | `Country` | Nominal | Negara tempat tinggal pelanggan |

**Business case.** Mahasiswa berperan sebagai data analyst baru di RetailKu. Dalam cerita, sekarang awal April 2011 dan manajer bertanya: *"Penjualan kuartal pertama tahun ini turun dibanding kuartal sebelumnya. Kenapa? Apa kita perlu khawatir?"*

**Tujuan pembelajaran.** Setelah pertemuan ini mahasiswa dapat:

1. Menjelaskan apa itu data science dan contoh penerapannya.
2. Membedakan peran Data Analyst, Data Scientist, Data Engineer, dan ML Engineer.
3. Menyebutkan enam tahap CRISP-DM.
4. Membedakan analitik descriptive, diagnostic, predictive, dan prescriptive.
5. Memuat data ke pandas dan mengenalinya dengan `head()`, `info()`, dan `describe()`.
6. Membuat grafik tren penjualan dan mengubah pertanyaan bisnis menjadi pertanyaan data.


## Lisensi dan sitasi

Dataset dilisensikan di bawah [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): boleh dibagikan dan diadaptasi untuk tujuan apa pun selama mencantumkan sumbernya.

Sitasi sesuai halaman UCI:

> Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
