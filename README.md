# Proyek Klasifikasi Gambar

Submission Dicoding untuk klasifikasi gambar menggunakan TensorFlow/Keras.

## Dataset

Proyek ini memakai CIFAR-10 sebagai dataset utama:

- 60.000 gambar.
- 10 kelas.
- Bukan dataset Rock Paper Scissors.
- Bukan dataset X-Ray.
- Split yang digunakan: 40.000 train, 10.000 validation, 10.000 test.

Jika ingin memakai dataset folder sendiri, letakkan gambar di `dataset/raw/<nama_kelas>/` lalu jalankan script dengan `--dataset folder`.

## Menjalankan

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python train_export.py --dataset cifar10 --epochs 40
```

Output model akan dibuat di folder:

```text
submission/
  saved_model/
  tflite/model.tflite
  tfjs_model/model.json
  notebook.ipynb
  README.md
  requirements.txt
```

Notebook `submission/notebook.ipynb` berisi alur yang sama dan dapat dijalankan dari atas sampai bawah.

## Training Lokal

Untuk menjalankan pipeline lokal dan menghasilkan struktur `submission/` sesuai ketentuan, lihat `RUN_LOCAL_GPU.md` dan jalankan `train_local_gpu.py`.

Hasil lokal terakhir menggunakan WSL2 GPU:

```text
train accuracy : 98.29%
test accuracy  : 91.16%
```

Untuk export ulang dari checkpoint tanpa training ulang, jalankan:

```bash
source .venv-wsl/bin/activate
python export_from_checkpoint.py
```
