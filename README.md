# DataMining
Laporan Proyek: Penggabungan dan Klasifikasi Data Scan Keamanan Jaringan
Deskripsi Singkat
Proyek ini merupakan implementasi pemrosesan data hasil scanning keamanan jaringan dari tiga sumber berbeda, yang kemudian digabungkan menjadi satu DataFrame. Dataset yang telah digabungkan digunakan sebagai dasar untuk pelatihan model klasifikasi menggunakan algoritma Decision Tree. Proyek ini mencakup proses preprocessing data, pemisahan fitur dan label, pelatihan model, evaluasi akurasi, serta visualisasi model pohon keputusan dan confusion matrix.

Struktur dan Penjelasan Kode
1. Import Library
python ```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
Kode ini mengimpor berbagai library yang dibutuhkan seperti:
pandas dan numpy untuk manipulasi data,
matplotlib dan seaborn untuk visualisasi,
scikit-learn untuk machine learning dan evaluasi model.

2. Membaca Data CSV
python ```
data1 = pd.read_csv('Recon OS Scan.csv')
data2 = pd.read_csv('Recon Port Scan.csv')
data3 = pd.read_csv('Recon Vulnerability Scan.csv')
```
Tiga file CSV dibaca yang berisi hasil pemindaian OS, pemindaian port, dan pemindaian kerentanan.

3. Preprocessing Data
python ```
dtRevisi = data2.iloc[:159025, :]
```
Data dari pemindaian port (data2) direduksi hingga 159.025 baris pertama untuk menjaga keselarasan dengan dataset lainnya.

4. Penggabungan DataFrame
python ```
DataJoin = pd.concat([data1, dtRevisi, data3], ignore_index=True)
```
Semua dataset digabung secara vertikal menjadi satu DataFrame (DataJoin). Penggunaan ignore_index=True mengatur ulang indeks baris agar tetap unik dan berurutan.

5. Pemisahan Fitur dan Label
python ```
x = DataJoin.iloc[:, 7:76]
y = DataJoin.iloc[:, 83:84]
```
Fitur (x) diambil dari kolom ke-7 hingga 75.
Label (y) diambil dari kolom ke-83. Pemilihan kolom ini mengasumsikan data label berada di kolom tersebut.

6. Penanganan Missing Values dan Pembagian Data
python ```
x = x.fillna(method='ffill')
y = y.fillna(method='ffill')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
```
Mengisi nilai yang hilang dengan metode forward-fill.
Membagi data menjadi 70% data latih dan 30% data uji.

7. Pelatihan Model Decision Tree
python ```
uji = DecisionTreeClassifier(criterion='entropy', splitter='random')
uji.fit(x_train, y_train)
y_pred = uji.predict(x_test)
```
Model pohon keputusan dibangun dengan kriteria entropy dan splitter='random'.
Model dilatih dengan data latih dan menghasilkan prediksi pada data uji.

8. Evaluasi Akurasi Model

python ```
accuracy = accuracy_score(y_test, y_pred)
```
Evaluasi menggunakan metrik akurasi antara label aktual dan prediksi.

9. Visualisasi Model Decision Tree
python ```
fig = plt.figure(figsize=(10, 7))
tree.plot_tree(uji, feature_names=x.columns.values, class_names=np.array(['Recon OS Scan', 'Recon Port Scan', 'Recon Vulnerability Scan']), filled=True)
plt.show()
```
Visualisasi struktur pohon keputusan secara grafis dengan fitur dan nama kelas yang sesuai.

10. Confusion Matrix
python ```
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sea.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
Menampilkan confusion matrix dalam bentuk heatmap untuk melihat performa klasifikasi dari masing-masing kelas.

Kesimpulan
Kode dataconcat.py ini merupakan pipeline analitik lengkap mulai dari pembacaan data keamanan jaringan, pengolahan dan penggabungan data, pembuatan model klasifikasi menggunakan Decision Tree, hingga visualisasi evaluasi model. Dengan pendekatan ini, dapat diperoleh pemahaman mendalam tentang kemampuan model dalam mengklasifikasikan jenis scan keamanan secara otomatis berdasarkan fitur yang tersedia. Akurasi dan visualisasi seperti pohon keputusan dan confusion matrix memberikan gambaran intuitif tentang kinerja model.

Catatan Tambahan
- File CSV eksternal perlu tersedia dalam direktori yang sama saat menjalankan script ini.
- Untuk pemodelan lebih lanjut, dapat dieksplorasi model klasifikasi lain seperti Random Forest, SVM, atau Neural Networks untuk membandingkan performa.
- Script ini dikembangkan secara otomatis melalui Google Colab dan kompatibel dengan Python 3 dan Scikit-learn.
