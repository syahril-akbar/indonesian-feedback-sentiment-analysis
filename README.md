# Implementasi K-Means dan Analisis Sentimen Kritik Saran Berbasis NLP pada Data Monev BBPSDMP Kominfo Makassar

Proyek ini adalah implementasi teknik Natural Language Processing (NLP) untuk melakukan analisis sentimen pada data kritik dan saran. Studi kasus yang diambil adalah data hasil monitoring dan evaluasi (Monev) dari kegiatan yang diselenggarakan oleh Balai Besar Pengembangan Sumber Daya Manusia dan Penelitian (BBPSDMP) Kominfo Makassar.

Tujuan utama dari proyek ini adalah untuk mengelompokkan ulasan secara otomatis ke dalam topik-topik yang relevan menggunakan algoritma **K-Means Clustering** dan kemudian membangun model klasifikasi sentimen menggunakan **Naive Bayes**.

---

## 📂 Alur Kerja Proyek

Alur kerja yang diimplementasikan dalam notebook `sentiment_analysis.ipynb` adalah sebagai berikut:

1.  **Pemuatan Data**: Memuat data ulasan dari file `kritik_saran.xlsx`.
2.  **Pra-pemrosesan Data**:
    *   Mengubah semua teks menjadi string.
    *   Menghilangkan *stopwords* (kata-kata umum seperti "yang", "di", "dan") dalam bahasa Indonesia menggunakan library Sastrawi.
    *   Mengubah teks menjadi representasi numerik menggunakan **TF-IDF Vectorization**.
3.  **Filter Relevansi**: Memisahkan ulasan yang dianggap relevan dan tidak relevan berdasarkan skor TF-IDF dan jumlah kata. Ulasan yang tidak relevan disimpan dalam `ulasan_tidak_relevan.xlsx`.
4.  **Clustering (K-Means)**:
    *   Menggunakan ulasan yang relevan, program menerapkan K-Means untuk menemukan pola dan mengelompokkan ulasan ke dalam 2 klaster utama.
    *   Hasil analisis kata kunci pada tiap klaster menghasilkan label:
        *   **Klaster 0: Kritik dan Harapan** (fokus pada fasilitas, ruangan, jaringan).
        *   **Klaster 1: Saran dan Pujian** (fokus pada materi, pengajar, dan apresiasi).
    *   Hasil akhir dari proses clustering disimpan dalam file `hasil_klaster_teks_asli.xlsx`.
5.  **Pelatihan Model Klasifikasi (Naive Bayes)**:
    *   Data yang sudah dilabeli dari hasil clustering digunakan untuk melatih model Naive Bayes.
    *   Tujuannya adalah agar model dapat memprediksi kategori sentimen dari ulasan baru secara otomatis.
6.  **Evaluasi Model**: Model dievaluasi menggunakan data uji dan mencapai **akurasi sekitar 89%**.
7.  **Penyimpanan Model**: Model Naive Bayes dan TF-IDF Vectorizer yang telah dilatih disimpan ke dalam file `Model_NB.pkl` untuk penggunaan di masa depan.

---

## 🛠️ Teknologi yang Digunakan

*   **Python 3**
*   **Jupyter Notebook**
*   **Scikit-learn**: Untuk implementasi TF-IDF, K-Means, Naive Bayes, dan evaluasi model.
*   **Pandas**: Untuk manipulasi dan analisis data.
*   **Sastrawi**: Untuk proses penghapusan *stopwords* bahasa Indonesia.
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
