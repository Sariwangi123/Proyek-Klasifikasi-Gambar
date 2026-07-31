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

Folder model pada submission ini sudah berisi hasil export final:

- SavedModel: `saved_model/saved_model.pb`
- TF-Lite: `tflite/model.tflite` dan `tflite/label.txt`
- TFJS: `tfjs_model/model.json` dan shard `.bin`

## Dataset

Dataset utama adalah CIFAR-10. Seluruh gambar CIFAR-10 digabung terlebih dahulu ke folder kelas masing-masing, lalu dibagi ulang secara manual menjadi train, validation, dan test set. Dataset ini memiliki 60.000 gambar dari 10 kelas dan tidak termasuk dataset Rock Paper Scissors maupun X-Ray.

Split data:

```text
train       : 48000 gambar
validation  : 6000 gambar
test        : 6000 gambar
```

## Menjalankan Notebook

1. Install dependensi dari `requirements.txt`.
2. Jalankan semua cell pada `notebook.ipynb`.
3. Pastikan output training, evaluasi test set, plot akurasi/loss, dan contoh inference TF-Lite terlihat di notebook.
4. Zip folder `submission` setelah folder model berhasil dibuat.

## Hasil Terakhir

Hasil evaluasi terakhir tersimpan di `metadata.json`.

```text
train accuracy : 95.95%
test accuracy  : 96.00%
inference      : predicted airplane, actual airplane
```
