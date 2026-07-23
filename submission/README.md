# Proyek Klasifikasi Gambar

Submission ini berisi notebook klasifikasi gambar menggunakan TensorFlow/Keras dengan dataset CIFAR-10.

## Isi Submission

```text
submission/
  tfjs_model/
  tflite/
  saved_model/
  notebook.ipynb
  README.md
  requirements.txt
```

Folder `saved_model`, `tflite`, dan `tfjs_model` akan terisi setelah notebook atau `train_export.py` dijalankan.

## Dataset

Dataset utama adalah CIFAR-10 yang otomatis diunduh oleh `tf.keras.datasets.cifar10`. Dataset ini memiliki 60.000 gambar dari 10 kelas dan tidak termasuk dataset Rock Paper Scissors maupun X-Ray.

Split data:

```text
train       : 40000 gambar
validation  : 10000 gambar
test        : 10000 gambar
```

## Menjalankan Notebook

1. Install dependensi dari `requirements.txt`.
2. Install converter TFJS dengan `pip install tensorflowjs==4.22.0 --no-deps`.
3. Jalankan semua cell pada `notebook.ipynb`.
4. Pastikan output training, evaluasi test set, plot akurasi/loss, dan contoh inference TF-Lite terlihat di notebook.
5. Zip folder `submission` setelah folder model berhasil dibuat.
