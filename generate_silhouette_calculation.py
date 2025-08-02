import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Muat data yang relevan
df = pd.read_excel('ulasan_relevan.xlsx', sheet_name="Sheet1")
df['ULASAN'] = df['ULASAN'].astype(str)

# Lakukan stemming dan penghapusan stopword (seperti di notebook)
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
indonesian_stopwords = stopword_factory.get_stop_words()
# Terapkan stemming pada kolom 'ULASAN'
# df['ULASAN'] = df['ULASAN'].apply(lambda x: stemmer.stem(x))

# Vektorisasi TF-IDF
vectorizer_tfidf = TfidfVectorizer(stop_words=indonesian_stopwords)
X_tfidf = vectorizer_tfidf.fit_transform(df['ULASAN'])

# Gunakan parameter terbaik
n_components = 50
n_clusters = 2

# Reduksi dimensi dengan SVD
svd = TruncatedSVD(n_components=n_components, random_state=0)
X_reduced = svd.fit_transform(X_tfidf)

# Klastering dengan K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_reduced)
df['Topik'] = labels

# Ambil indeks sampel dari setiap klaster
cluster_0_indices = np.where(labels == 0)[0]
cluster_1_indices = np.where(labels == 1)[0]

# Ambil 3 sampel pertama dari setiap klaster untuk demonstrasi
sample_indices_c0 = cluster_0_indices[:3]
sample_indices_c1 = cluster_1_indices[:3]
sample_indices = np.concatenate([sample_indices_c0, sample_indices_c1])

# Dapatkan titik data yang sesuai
sample_data = X_reduced[sample_indices]
sample_labels = labels[sample_indices]
sample_ulasan = df['ULASAN'].iloc[sample_indices].values

# Hitung jarak berpasangan (pairwise distances)
distance_matrix = pairwise_distances(sample_data)

# --- Cetak hasil untuk format markdown ---
print("#### a. Perhitungan Manual Silhouette Score")
print("\nSkor Silhouette untuk satu titik data (`i`) dihitung dengan rumus:")
print("\n**s(i) = (b(i) - a(i)) / max(a(i), b(i))**")
print("\nDi mana:")
print("-   `a(i)`: Jarak rata-rata dari titik `i` ke semua titik lain di dalam klaster yang sama (ukuran kohesi).")
print("-   `b(i)`: Jarak rata-rata dari titik `i` ke semua titik di klaster terdekat berikutnya (ukuran separasi).")
print("\n**Contoh Perhitungan Berdasarkan Data Aktual:**")
print("Untuk demonstrasi, kita akan mengambil 3 sampel ulasan dari masing-masing klaster yang dihasilkan oleh model. Kita akan menghitung skor silhouette untuk **P1**.")
print("\n**Sampel Data:**")
for i, (idx, label, ulasan) in enumerate(zip(sample_indices, sample_labels, sample_ulasan)):
    print(f"- **P{i+1}**: Ulasan `\"{ulasan[:60]}...\"` (Klaster {label})")

print("\n**Matriks Jarak (Euclidean) Antar Sampel:**")
print("```")
print("        P1      P2      P3      P4      P5      P6")
for i in range(len(sample_data)):
    print(f"P{i+1}  {np.array2string(distance_matrix[i], formatter={'float_kind':lambda x: '%.4f' % x})}")
print("```")

# Perhitungan manual untuk P1
p1_index_in_sample = 0
# Jarak di dalam klaster yang sama (Klaster 0): P1 -> P2, P1 -> P3
a_p1_distances = [distance_matrix[p1_index_in_sample, 1], distance_matrix[p1_index_in_sample, 2]]
a_p1 = np.mean(a_p1_distances)

# Jarak ke klaster terdekat berikutnya (Klaster 1): P1 -> P4, P5, P6
b_p1_distances = [distance_matrix[p1_index_in_sample, 3], distance_matrix[p1_index_in_sample, 4], distance_matrix[p1_index_in_sample, 5]]
b_p1 = np.mean(b_p1_distances)

# Skor silhouette untuk P1
s_p1 = (b_p1 - a_p1) / max(a_p1, b_p1)

print("\n**Langkah-langkah Perhitungan untuk P1 (dari Klaster 0):**")
print(f"1.  **Hitung `a(P1)` (Kohesi):**")
print(f"    Jarak rata-rata dari P1 ke titik lain di Klaster 0 (P2 dan P3).")
print(f"    - Jarak(P1, P2) = {a_p1_distances[0]:.4f}")
print(f"    - Jarak(P1, P3) = {a_p1_distances[1]:.4f}")
print(f"    - `a(P1)` = ({a_p1_distances[0]:.4f} + {a_p1_distances[1]:.4f}) / 2 = **{a_p1:.4f}**")
print("\n2.  **Hitung `b(P1)` (Separasi):**")
print(f"    Jarak rata-rata dari P1 ke semua titik di klaster terdekat (Klaster 1), yaitu P4, P5, dan P6.")
print(f"    - Jarak(P1, P4) = {b_p1_distances[0]:.4f}")
print(f"    - Jarak(P1, P5) = {b_p1_distances[1]:.4f}")
print(f"    - Jarak(P1, P6) = {b_p1_distances[2]:.4f}")
print(f"    - `b(P1)` = ({b_p1_distances[0]:.4f} + {b_p1_distances[1]:.4f} + {b_p1_distances[2]:.4f}) / 3 = **{b_p1:.4f}**")
print("\n3.  **Hitung Skor Silhouette `s(P1)`:**")
print(f"    `s(P1)` = (`b(P1)` - `a(P1)`) / max(`a(P1)`, `b(P1)`)")
print(f"    `s(P1)` = ({b_p1:.4f} - {a_p1:.4f}) / max({a_p1:.4f}, {b_p1:.4f})")
print(f"    `s(P1)` = {b_p1 - a_p1:.4f} / {max(a_p1, b_p1):.4f} = **{s_p1:.4f}**")
print("\nSkor Silhouette keseluruhan (0.1212) adalah rata-rata dari skor `s(i)` yang dihitung untuk *semua* 2307 titik data dalam himpunan data relevan.")

