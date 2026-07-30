# Menjalankan Training Lokal Dengan GPU

Output akhir mengikuti struktur submission Dicoding:

```text
submission/
  tfjs_model/
    group1-shard1of1.bin
    model.json
  tflite/
    model.tflite
    label.txt
  saved_model/
    saved_model.pb
    variables/
  Template_Submission_Akhir.ipynb
  README.md
  requirements.txt
```

## Catatan GPU Windows

TensorFlow 2.17 pada Windows native umumnya berjalan di CPU. Untuk memakai GPU NVIDIA/RTX, jalur yang disarankan adalah menjalankan script ini di WSL2 atau Docker Linux dengan GPU support.

## Jalur Disarankan: WSL2 Tanpa Docker

Di terminal Ubuntu WSL:

```bash
cd "/mnt/d/Pelatihan Dicoding/Proyek Klasifikasi Gambar"
bash setup_wsl_gpu.sh
```

Kalau output `check_gpu.py` menampilkan GPU, jalankan training:

```bash
source .venv-wsl/bin/activate
python train_local_gpu.py
```

## Jalankan Di Windows Native

Pastikan file dataset tersedia:

```text
submission/cifar-10-python.tar.gz
```

Lalu jalankan:

```powershell
.\.venv\Scripts\python.exe train_local_gpu.py
```

Di awal output, cek baris:

```text
GPUs: [...]
```

Jika kosong (`GPUs: []`), TensorFlow tidak sedang memakai VGA/GPU.
