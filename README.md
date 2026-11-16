# AI Model Experiment & Evaluation

**ReWork AI Engineering Bootcamp — Individual Assignment**  
**Use case:** Klasifikasi sentimen ulasan pelanggan e-commerce (`positif` / `negatif`)  
**Pendekatan:** TF-IDF + Logistic Regression (Scikit-learn) vs. Gemini API (zero-shot prompting)

## 1. Problem Statement dan Objective

Sebuah tim produk e-commerce membutuhkan fitur untuk mengklasifikasikan sentimen ulasan pelanggan secara otomatis. Eksperimen ini membandingkan dua pendekatan AI pada **test set yang identik** untuk menentukan pendekatan yang paling menjanjikan bagi pengembangan berikutnya.

**Objective:**

1. Melatih model Machine Learning klasik menggunakan **Scikit-learn** untuk memprediksi sentimen ulasan.
2. Menjalankan klasifikasi sentimen menggunakan **Gemini API** tanpa training tambahan (zero-shot prompting).
3. Membandingkan **Accuracy, Precision, Recall, F1-Score (macro)** dan **confusion matrix**.
4. Menganalisis performa, kesalahan prediksi, effort implementasi, kecepatan, biaya, dan keterbatasan kedua pendekatan.
5. Menyusun rekomendasi *technical approach* berdasarkan bukti eksperimen.

**Input:** kolom `review_text` (ulasan berbahasa Indonesia).  
**Target/label:** kolom `sentiment` dengan dua kelas: `positif` dan `negatif`.

**Ruang lingkup:** eksperimen awal berbasis dataset terlampir, bukan validasi model untuk penerapan produksi. Kategori `netral` tidak tersedia pada dataset.

## 2. Dataset dan Asumsi

Dataset utama berada di `data/customer_reviews_sentiment.csv`; file sumber Excel `data/customer_reviews_sentiment.csv.xlsx` turut disimpan sebagai referensi. Dataset asli tidak diubah selama eksperimen.

| Keterangan | Jumlah / Detail |
|---|---:|
| Total baris | 200 |
| Jumlah kolom | 4 |
| Ulasan positif | 110 |
| Ulasan negatif | 90 |
| Teks ulasan unik | 40 |
| Data training | 158 baris / 32 teks unik |
| Data testing | 42 baris / 8 teks unik |
| Distribusi testing | 24 negatif dan 18 positif |

**Kolom dataset:** `review_id`, `product_name`, `review_text`, dan `sentiment`.

### Strategi train/test split

Karena terdapat 200 baris tetapi hanya **40 teks ulasan unik**, pembagian acak langsung pada baris berisiko menempatkan kalimat yang sama di training dan test set (*data leakage*). Untuk mengurangi risiko ini:

1. Ambil kumpulan `review_text` yang unik beserta labelnya.
2. Bagi teks unik menjadi **80% training / 20% testing** menggunakan `train_test_split(test_size=0.20, random_state=42, stratify=...)`.
3. Kembalikan seluruh baris dataset asli ke kelompok training/test berdasarkan teks ulasannya.
4. Pastikan tidak ada teks ulasan identik yang muncul di kedua kelompok.
5. Gunakan **42 baris test yang sama** untuk mengevaluasi Scikit-learn dan Gemini.

> **Catatan:** Walaupun jumlah baris testing adalah 42, jumlah **teks berbeda** yang diuji hanya **8**. Oleh karena itu, hasil evaluasi harus ditafsirkan secara hati-hati dan tidak boleh dianggap sudah merepresentasikan performa pada ulasan produksi.

## 3. Metodologi Eksperimen

### 3.1 Pendekatan 1 — Machine Learning Klasik

- **Feature extraction:** `TfidfVectorizer(ngram_range=(1, 2), lowercase=True)`; menghasilkan **329 fitur** pada training set.
- **Model:** `LogisticRegression(max_iter=1000, random_state=42)`.
- **Training:** hanya menggunakan data training; TF-IDF di-*fit* hanya pada training set.
- **Inference:** `clf.predict()` untuk 42 baris test.
- **Waktu training tercatat:** sekitar **0,0149 detik** pada sesi notebook yang dilampirkan (bukan waktu inferensi total).

### 3.2 Pendekatan 2 — Gemini API

- **Model yang dipanggil dalam notebook:** `gemini-3.5-flash-lite`.
- **Teknik:** *zero-shot prompting*, tanpa training ulang.
- **Instruksi prompt:** tentukan apakah ulasan `positif` atau `negatif`, dan keluarkan hanya satu label.
- **Temperature:** `0.0`, untuk mengurangi variasi keluaran.
- **Validasi respons:** hasil dinormalisasi menjadi label `positif` / `negatif`; respons yang tidak valid ditolak.
- **Optimasi:** cache prediksi untuk teks identik sehingga tidak perlu mengulang request API.
- **Pelaksanaan dalam notebook:** **8 panggilan API baru** untuk 8 teks unik pada test set; kemudian hasilnya dipetakan kembali ke 42 baris test.
- **Waktu loop tercatat:** **33,29 detik** termasuk jeda terprogram (`sleep(3)`) serta waktu pemrosesan/cache. Angka ini **bukan latency API murni**.

**Keadilan perbandingan:** Kedua pendekatan menggunakan label asli yang sama untuk menghitung metrik evaluasi, dan label test tidak dikirimkan ke Gemini dalam prompt.

## 4. Hasil Evaluasi Model

Hasil berikut berasal dari eksekusi notebook dan file `evaluation_summary.csv` yang dilampirkan.

### 4.1 Perbandingan metrik

| Metrik | TF-IDF + Logistic Regression | Gemini API |
|---|---:|---:|
| **Accuracy** | 69,05% | **100,00%** |
| **Precision (macro)** | 79,03% | **100,00%** |
| **Recall (macro)** | 72,92% | **100,00%** |
| **F1-Score (macro)** | 68,16% | **100,00%** |
| Prediksi benar | 29 dari 42 | **42 dari 42** |
| Prediksi salah | 13 dari 42 | **0 dari 42** |

**Interpretasi:**

- **Accuracy:** Gemini memperoleh **100%** dibandingkan **69,05%** untuk model klasik (selisih **30,95 poin persentase**) pada test set ini.
- **Precision (macro):** Rata-rata ketepatan prediksi per kelas lebih tinggi pada Gemini (**100% vs. 79,03%**).
- **Recall (macro):** Gemini mengenali seluruh sampel dari kedua kelas pada test set ini (**100% vs. 72,92%**).
- **F1-Score (macro):** Gemini menghasilkan **100%** dibandingkan **68,16%** untuk model klasik (selisih **31,84 poin persentase**), menunjukkan keseimbangan precision dan recall yang lebih baik **dalam pengujian ini**.

Nilai `macro` berarti setiap kelas diberi bobot sama ketika nilai precision, recall, dan F1 dirata-ratakan.

### 4.2 Confusion matrix

Urutan label yang digunakan dalam notebook adalah **`[negatif, positif]`**. **Baris = label aktual; kolom = label prediksi.**

**TF-IDF + Logistic Regression**

| Aktual \\ Prediksi | Negatif | Positif |
|---|---:|---:|
| **Negatif** | 11 | 13 |
| **Positif** | 0 | 18 |

- Benar: **11 negatif** dan **18 positif**.
- Salah: **13 ulasan negatif diprediksi sebagai positif**.
- *Recall* kelas negatif = **11/24 = 45,83%** (ditampilkan sekitar **0,46** pada `classification_report`).
- *Recall* kelas positif = **18/18 = 100%**.

**Gemini API**

| Aktual \\ Prediksi | Negatif | Positif |
|---|---:|---:|
| **Negatif** | 24 | 0 |
| **Positif** | 0 | 18 |

- Seluruh **24 ulasan negatif** dan **18 ulasan positif** berhasil diprediksi benar pada test set ini.
- Tidak ditemukan kesalahan klasifikasi pada pengujian yang dilaporkan.

### 4.3 Error Analysis

Model klasik membuat **13 prediksi salah**, yang berasal dari **2 teks ulasan negatif unik**; Gemini mengklasifikasikan keduanya sebagai negatif dengan benar:

| Contoh ulasan | Label aktual | Prediksi Scikit-learn | Prediksi Gemini |
|---|---|---|---|
| "Respon penjual lambat dan terkesan tidak peduli komplain." | negatif | positif | negatif |
| "Ukuran dan warna berbeda dari pesanan, sangat kecewa." | negatif | positif | negatif |

**Implikasi bisnis:** Kesalahan membaca keluhan sebagai sentimen positif dapat membuat tim e-commerce melewatkan masalah pelanggan. Oleh sebab itu, **recall kelas negatif** perlu menjadi perhatian khusus, bukan hanya accuracy keseluruhan.

## 5. Analisis Trade-off

| Aspek | TF-IDF + Logistic Regression | Gemini API (zero-shot) |
|---|---|---|
| **Kualitas pada test set** | Accuracy 69,05%; F1-macro 68,16% | Accuracy dan F1-macro 100% |
| **Effort implementasi** | Perlu ekstraksi fitur, training, evaluasi, dan kemungkinan training ulang | Tidak perlu training; perlu prompt, API key, validasi respons, dan penanganan error |
| **Kecepatan** | Training tercatat 0,0149 detik; inferensi lokal tanpa request jaringan | Loop API 33,29 detik termasuk jeda buatan; latency request murni belum terukur |
| **Biaya** | Tidak ada biaya per panggilan API, tetapi ada biaya komputasi serta pemeliharaan | Berpotensi ada biaya penggunaan API/token; biaya aktual **tidak dicatat** |
| **Internet & ketersediaan** | Inferensi dapat berjalan secara lokal | Membutuhkan koneksi dan layanan API; dipengaruhi kuota/rate limit |
| **Privasi & kontrol data** | Pemrosesan dapat dilakukan di infrastruktur sendiri | Pengiriman data ke penyedia eksternal perlu ditinjau dari aspek privasi dan keamanan |

**Kesimpulan trade-off:** Gemini unggul pada metrik klasifikasi dalam eksperimen ini, sedangkan model klasik memiliki keunggulan potensial pada inferensi lokal, kontrol infrastruktur, dan ketiadaan tarif API per request. Angka waktu di atas **tidak dapat dibandingkan langsung** karena berasal dari proses berbeda (training klasik vs. loop request Gemini yang mencakup jeda).

## 6. Limitation / Keterbatasan

1. **Dataset sangat terbatas:** hanya **40 teks unik** dari 200 baris. Test set berisi **8 teks unik**, sehingga hasil 100% Gemini tidak menjamin hasil serupa pada data pelanggan baru.
2. **Variasi bahasa dan sentimen belum representatif:** belum cukup menguji teks ambigu, netral, sarkasme, typo, campuran bahasa, bahasa gaul, dan ulasan dengan sentimen campuran. Dataset hanya menyediakan dua label.
3. **Model klasik cenderung salah mengenali sebagian keluhan:** pada test set, **13 dari 24 baris negatif** salah diklasifikasikan sebagai positif. Dataset training yang kecil dapat membatasi generalisasi pola kata TF-IDF.
4. **Gemini bergantung pada prompt dan layanan eksternal:** hasil dapat berubah jika prompt, versi model, atau kondisi layanan berubah; ada risiko rate limit dan gangguan API.
5. **Biaya serta latency produksi belum diukur:** notebook belum mendokumentasikan token usage, pengeluaran API aktual, atau latency per request tanpa jeda buatan.
6. **Belum ada validasi pada dataset independen yang lebih besar:** eksperimen ini adalah *proof of concept*, bukan pengujian kesiapan produksi.

## 7. Technical Recommendation

### 7.1 Rekomendasi utama

**Gemini API dengan zero-shot prompting direkomendasikan sebagai kandidat utama untuk tahap pilot / proof of concept berikutnya**, **bukan langsung untuk deployment produksi**.

Alasan rekomendasi:

1. **Performa terbaik pada test set yang sama:** Gemini memperoleh **42/42 prediksi benar**, sementara TF-IDF + Logistic Regression memperoleh **29/42**.
2. **Lebih baik mendeteksi keluhan:** Gemini mengidentifikasi **24/24 ulasan negatif** dengan benar; model klasik hanya **11/24**.
3. **Proses awal lebih sederhana:** Gemini tidak memerlukan training dataset, sehingga bisa segera diuji dengan prompting dan API.
4. **Keputusan tetap bersyarat:** Keunggulan ini hanya terbukti pada **8 teks unik pada test set**; diperlukan validasi, pengukuran biaya, latency, serta pemeriksaan privasi sebelum keputusan produksi.

### 7.2 Rencana tindak lanjut sebelum implementasi produksi

1. **Perluas data pengujian** dengan lebih banyak ulasan unik dan representatif (negatif, positif, ambigu, sarkasme, typo, serta variasi bahasa); jaga agar teks identik tidak bocor antar kelompok train/test.
2. **Ukur kinerja operasional Gemini**: latency per request tanpa jeda terprogram, error/rate limit, pemakaian token, dan biaya estimasi per 1.000 atau 10.000 ulasan.
3. **Periksa keamanan dan privasi data** sebelum mengirimkan ulasan pelanggan ke layanan eksternal; tangani informasi pribadi bila ada.
4. **Tentukan ambang kelayakan bisnis**, terutama target minimum recall untuk sentimen negatif, F1, batas biaya, dan waktu respons.
5. **Pertahankan Scikit-learn sebagai baseline** dan evaluasi kembali setelah dataset diperbesar. Strategi hybrid lokal + API dapat dieksplorasi, tetapi **belum diuji dalam assignment ini**.

**Kesimpulan:** Berdasarkan hasil notebook, Gemini adalah pilihan paling menjanjikan untuk **pengembangan lanjutan**, sementara Scikit-learn menjadi baseline dan alternatif untuk kebutuhan inferensi lokal. Keputusan produksi harus menunggu pengujian pada data yang lebih beragam dan bukti biaya/operasional yang memadai.

## 8. Struktur Project dan Output

```text
model-experiment-assignment/
├── data/
│   ├── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   ├── evaluation_summary.csv            # hasil setelah notebook dijalankan
│   ├── test_predictions.csv               # hasil setelah notebook dijalankan
│   ├── confusion_matrix_classic.png       # hasil setelah notebook dijalankan
│   ├── confusion_matrix_gemini.png        # hasil setelah notebook dijalankan
│   ├── model_comparison_summary.png       # hasil setelah notebook dijalankan
│   └── gemini_prediction_cache.json       # cache yang dibuat saat eksekusi
├── README.md
└── requirements.txt
```