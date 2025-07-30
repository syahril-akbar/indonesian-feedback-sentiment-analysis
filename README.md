# Implementasi K-Means dan Analisis Sentimen Kritik Saran Berbasis NLP pada Data Monev BBPSDMP Kominfo Makassar

Proyek ini adalah implementasi teknik Natural Language Processing (NLP) untuk melakukan analisis sentimen pada data kritik dan saran. Studi kasus yang diambil adalah data hasil monitoring dan evaluasi (Monev) dari kegiatan yang diselenggarakan oleh Balai Besar Pengembangan Sumber Daya Manusia dan Penelitian (BBPSDMP) Kominfo Makassar.

Tujuan utama dari proyek ini adalah untuk mengelompokkan ulasan secara otomatis ke dalam topik-topik yang relevan menggunakan algoritma **K-Means Clustering** dan kemudian membangun model klasifikasi sentimen menggunakan **Naive Bayes**.

---

## 📂 Alur Kerja Proyek

Alur kerja yang diimplementasikan dalam notebook `sentiment_analysis.ipynb` adalah sebagai berikut:

1.  **Pemuatan Data**: Memuat data ulasan dari file `kritik_saran.xlsx`.
2.  **Pra-pemrosesan Data**:
    *   Mengubah semua teks menjadi string.
    *   **Stemming**: Mengubah kata menjadi bentuk dasarnya (misal: "mempelajari" -> "ajar") menggunakan Sastrawi.
    *   **Stopword Removal**: Menghilangkan kata-kata umum (misal: "yang", "di", "dan") menggunakan Sastrawi.
    *   **TF-IDF Vectorization**: Mengubah teks menjadi representasi numerik yang merefleksikan pentingnya sebuah kata dalam dokumen.
3.  **Filter Relevansi**: Memisahkan ulasan yang dianggap relevan dan tidak relevan berdasarkan skor TF-IDF dan jumlah kata. Ulasan yang relevan disimpan di `ulasan_relevan.xlsx` dan yang tidak relevan di `ulasan_tidak_relevan.xlsx`.
4.  **Clustering (K-Means)**:
    *   **Reduksi Dimensi**: Menggunakan **TruncatedSVD** untuk mengurangi kompleksitas data TF-IDF sebelum clustering.
    *   **K-Means**: Menerapkan K-Means pada data yang telah direduksi untuk menemukan pola dan mengelompokkan ulasan ke dalam 2 klaster utama.
    *   Hasil analisis kata kunci pada tiap klaster menghasilkan label:
        *   **Klaster 0: Kritik dan Harapan** (kata kunci: 'lebih', 'baik', 'kurang', 'ruang', 'makan', 'fasilitas').
        *   **Klaster 1: Saran dan Pujian** (kata kunci: 'materi', 'ajar', 'paham', 'jelas', 'mudah', 'sangat').
    *   Hasil akhir dari proses clustering disimpan dalam file `hasil_klaster_teks_asli.xlsx`.
5.  **Pelatihan Model Klasifikasi (Naive Bayes)**:
    *   Data berlabel dari hasil clustering digunakan untuk melatih model klasifikasi **Multinomial Naive Bayes**.
    *   Tujuannya adalah agar model dapat memprediksi kategori sentimen dari ulasan baru secara otomatis.
6.  **Evaluasi Model**: Model dievaluasi menggunakan data uji dan mencapai **akurasi sekitar 91%**.
7.  **Penyimpanan Model**: Model Naive Bayes dan TF-IDF Vectorizer yang telah dilatih disimpan ke dalam file `Model_NB.pkl` menggunakan `joblib` untuk penggunaan di masa depan.

---

## 🛠️ Teknologi yang Digunakan

*   **Python 3**
*   **Jupyter Notebook**
*   **Scikit-learn**: Untuk implementasi TF-IDF, TruncatedSVD, K-Means, Naive Bayes, dan evaluasi model.
*   **Pandas**: Untuk manipulasi dan analisis data.
*   **Sastrawi**: Untuk proses *stemming* dan penghapusan *stopwords* bahasa Indonesia.
*   **Numpy**: Untuk komputasi numerik.
*   **Matplotlib**: Untuk visualisasi data.
*   **Joblib**: Untuk menyimpan dan memuat model.

---

## 🚀 Cara Menjalankan

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone Repositori**
    ```bash
    git clone https://github.com/syahril-akbar/indonesian-feedback-sentiment-analysis.git
    cd indonesian-feedback-sentiment-analysis
    ```

2.  **Buat Virtual Environment (Direkomendasikan)**
    ```bash
    python -m venv venv
    ```
    *   Pada Windows, aktifkan dengan:
        ```bash
        .\venv\Scripts\activate
        ```
    *   Pada macOS/Linux, aktifkan dengan:
        ```bash
        source venv/bin/activate
        ```

3.  **Instal Dependensi**
    Pastikan Anda memiliki file `requirements.txt` yang berisi semua library yang dibutuhkan, lalu jalankan:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Jupyter Notebook**
    Buka dan jalankan sel-sel kode di dalam file `sentiment_analysis.ipynb` untuk melihat keseluruhan proses.
    ```bash
    jupyter notebook sentiment_analysis.ipynb
    ```

---

## 📁 Struktur Folder

```
.
├── .env
├── .gitignore
├── hasil_klaster_teks_asli.xlsx  # Output: Ulasan dengan label klaster
├── kritik_saran.xlsx             # Input: Data ulasan mentah
├── Model_NB.pkl                  # Output: Model yang sudah dilatih
├── README.md                     # Dokumentasi proyek
├── requirements.txt              # Daftar dependensi
├── sentiment_analysis.ipynb      # Notebook utama untuk analisis
├── ulasan_relevan.xlsx           # Output: Ulasan yang relevan
└── ulasan_tidak_relevan.xlsx     # Output: Ulasan yang tidak relevan
```

---

## 👨‍💻 Kontributor

*   **Syahril Akbar** - *Initial work* - [https://github.com/syahril-akbar](https://github.com/syahril-akbar)
