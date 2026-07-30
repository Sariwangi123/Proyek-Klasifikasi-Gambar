import tensorflow as tf


print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("GPU belum terbaca oleh TensorFlow.")

print("GPU siap dipakai TensorFlow.")
