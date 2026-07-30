import json
import pickle
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


SEED = 42
BATCH_SIZE = 64
WARMUP_EPOCHS = 10
FINE_TUNE_EPOCHS = 40
TARGET_SIZE = (96, 96)
DATASET_ARCHIVE = Path("submission/cifar-10-python.tar.gz")
WORK_DIR = Path("local_training")
EXTRACT_DIR = WORK_DIR / "cifar10_data"
SUBMISSION_DIR = Path("submission")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
BEST_MODEL_PATH = SUBMISSION_DIR / "best_model.keras"
TRAINING_LOG_PATH = SUBMISSION_DIR / "training_log.csv"


def print_devices():
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))


def extract_dataset():
    if not DATASET_ARCHIVE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_ARCHIVE}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    cifar_dir = EXTRACT_DIR / "cifar-10-batches-py"
    if not cifar_dir.exists():
        with tarfile.open(DATASET_ARCHIVE, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
    return cifar_dir


def load_cifar_batch(batch_path):
    with open(batch_path, "rb") as file:
        batch = pickle.load(file, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch["labels"])
    return images, labels


def load_dataset():
    cifar_dir = extract_dataset()
    train_images = []
    train_labels = []
    for batch_name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]:
        images, labels = load_cifar_batch(cifar_dir / batch_name)
        train_images.append(images)
        train_labels.append(labels)

    x_train_full = np.concatenate(train_images)
    y_train_full = np.concatenate(train_labels).reshape(-1, 1)
    x_test, y_test = load_cifar_batch(cifar_dir / "test_batch")
    y_test = y_test.reshape(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_full.reshape(-1),
    )
    return x_train_full, y_train_full, x_train, y_train, x_val, y_val, x_test, y_test


def make_dataset(images, labels, shuffle=False):
    labels = labels.reshape(-1).astype("int64")
    ds = tf.data.Dataset.from_tensor_slices((images.astype("float32"), labels))
    if shuffle:
        ds = ds.shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Resizing(TARGET_SIZE[0], TARGET_SIZE[1]),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomZoom(0.10),
        base_model,
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="custom_conv2d_head"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), name="custom_max_pooling_head"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def find_base_model(model):
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            return layer
    raise ValueError("EfficientNet base model tidak ditemukan di model.")


def last_logged_epoch():
    if not TRAINING_LOG_PATH.exists():
        return None
    lines = [line.strip() for line in TRAINING_LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1:
        return None
    return int(lines[-1].split(",", 1)[0])


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUBMISSION_DIR / "accuracy_loss_plot.png", dpi=160)
    plt.close()


def export_models(best_model):
    saved_model_dir = SUBMISSION_DIR / "saved_model"
    tflite_dir = SUBMISSION_DIR / "tflite"
    tfjs_dir = SUBMISSION_DIR / "tfjs_model"

    for path in [saved_model_dir, tflite_dir, tfjs_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    tf.saved_model.save(best_model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    (tflite_dir / "model.tflite").write_bytes(tflite_model)
    (tflite_dir / "label.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tensorflowjs.converters.converter",
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                str(saved_model_dir),
                str(tfjs_dir),
            ],
            check=True,
        )
    except Exception as exc:
        print("TFJS export gagal. Install tensorflowjs lalu jalankan ulang export.")
        print(exc)


def run_tflite_inference(x_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=str(SUBMISSION_DIR / "tflite" / "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample_index = 0
    sample_image = x_test[sample_index].astype("float32")
    sample_batch = np.expand_dims(sample_image, axis=0).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], sample_batch)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
    actual_class = CLASS_NAMES[int(y_test[sample_index][0])]
    return predicted_class, actual_class, float(np.max(prediction))


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print_devices()

    x_train_full, y_train_full, x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    print("Total gambar:", len(x_train_full) + len(x_test))
    print("Train:", len(x_train))
    print("Validation:", len(x_val))
    print("Test:", len(x_test))
    print("Jumlah kelas:", len(CLASS_NAMES))

    train_ds = make_dataset(x_train, y_train, shuffle=True)
    val_ds = make_dataset(x_val, y_val)
    test_ds = make_dataset(x_test, y_test)

    resume_epoch = last_logged_epoch()
    if BEST_MODEL_PATH.exists() and resume_epoch is not None:
        print(f"Resume dari {BEST_MODEL_PATH} setelah epoch {resume_epoch}.")
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        base_model = find_base_model(model)
        base_model.trainable = True
        for layer in base_model.layers[:-60]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_warmup = None
    else:
        model, base_model = build_model()
        history_warmup = "pending"

    model.summary()
    print("Menggunakan Sequential:", isinstance(model, tf.keras.Sequential))
    print("Layer Conv2D eksplisit:", [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)])
    print("Layer Pooling eksplisit:", [layer.name for layer in model.layers if isinstance(layer, (tf.keras.layers.MaxPooling2D, tf.keras.layers.GlobalAveragePooling2D))])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(TRAINING_LOG_PATH, append=BEST_MODEL_PATH.exists()),
    ]

    if history_warmup == "pending":
        history_warmup = model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=callbacks)

        base_model.trainable = True
        for layer in base_model.layers[:-60]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        initial_epoch = len(history_warmup.history["loss"])
    else:
        initial_epoch = (resume_epoch or WARMUP_EPOCHS - 1) + 1

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    history = {}
    if history_warmup:
        for key in history_warmup.history:
            history[key] = history_warmup.history[key] + history_finetune.history.get(key, [])
    else:
        history = history_finetune.history

    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    train_loss, train_acc = best_model.evaluate(train_ds)
    test_loss, test_acc = best_model.evaluate(test_ds)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_history(history)
    export_models(best_model)
    predicted_class, actual_class, confidence = run_tflite_inference(x_test, y_test)
    print("Prediksi:", predicted_class)
    print("Aktual:", actual_class)
    print("Confidence:", confidence)

    metadata = {
        "dataset": "CIFAR-10",
        "total_images": int(len(x_train_full) + len(x_test)),
        "classes": CLASS_NAMES,
        "train_images": int(len(x_train)),
        "validation_images": int(len(x_val)),
        "test_images": int(len(x_test)),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "tflite_inference_example": {
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "confidence": confidence,
        },
    }
    (SUBMISSION_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
