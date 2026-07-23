import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def collect_images(data_dir: Path):
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    image_paths = []
    labels = []
    class_names = []

    for index, class_dir in enumerate(class_dirs):
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            class_names.append(class_dir.name)
            image_paths.extend(files)
            labels.extend([len(class_names) - 1] * len(files))

    if len(class_names) < 2:
        raise ValueError("Dataset harus memiliki minimal 2 kelas.")
    if len(image_paths) < 1000:
        raise ValueError(f"Dataset hanya berisi {len(image_paths)} gambar. Minimal 1000 gambar.")

    return np.array([str(p) for p in image_paths]), np.array(labels), class_names


def make_image_folder_dataset(paths, labels, image_size, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_array_dataset(images, labels, batch_size, shuffle=False):
    labels = labels.reshape(-1).astype("int64")
    ds = tf.data.Dataset.from_tensor_slices((images.astype("float32") / 255.0, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_cifar10(batch_size):
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full.reshape(-1),
    )
    return {
        "class_names": CIFAR10_CLASSES,
        "image_size": (32, 32),
        "train_ds": make_array_dataset(x_train, y_train, batch_size, shuffle=True),
        "val_ds": make_array_dataset(x_val, y_val, batch_size),
        "test_ds": make_array_dataset(x_test, y_test, batch_size),
        "train_images": len(x_train),
        "validation_images": len(x_val),
        "test_images": len(x_test),
        "total_images": len(x_train_full) + len(x_test),
        "sample_image": x_test[0],
        "sample_label": int(y_test[0][0]),
        "dataset_name": "CIFAR-10",
    }


def load_image_folder(data_dir, image_size, batch_size):
    paths, labels, class_names = collect_images(data_dir)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )
    return {
        "class_names": class_names,
        "image_size": image_size,
        "train_ds": make_image_folder_dataset(train_paths, train_labels, image_size, batch_size, shuffle=True),
        "val_ds": make_image_folder_dataset(val_paths, val_labels, image_size, batch_size),
        "test_ds": make_image_folder_dataset(test_paths, test_labels, image_size, batch_size),
        "train_images": len(train_paths),
        "validation_images": len(val_paths),
        "test_images": len(test_paths),
        "total_images": len(paths),
        "sample_image": test_paths[0],
        "sample_label": int(test_labels[0]),
        "dataset_name": f"Image folder: {data_dir}",
    }


def build_model(image_size, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def plot_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_loss_plot.png", dpi=160)
    plt.close()


def export_models(model, output_dir: Path, image_size):
    saved_model_dir = output_dir / "saved_model"
    tflite_dir = output_dir / "tflite"
    tfjs_dir = output_dir / "tfjs_model"

    for path in [saved_model_dir, tflite_dir, tfjs_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    tf.saved_model.save(model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (tflite_dir / "model.tflite").write_bytes(tflite_model)

    subprocess.run(
        [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            str(saved_model_dir),
            str(tfjs_dir),
        ],
        check=True,
    )


def prepare_sample_image(sample_image, image_size):
    if isinstance(sample_image, np.ndarray):
        image = tf.image.resize(sample_image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, "cifar10_test_sample"

    image = tf.io.read_file(str(sample_image))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, str(sample_image)


def run_inference_example(tflite_path: Path, sample_image, class_names, image_size):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image, image_name = prepare_sample_image(sample_image, image_size)
    image = tf.expand_dims(image, axis=0).numpy().astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_index = int(np.argmax(prediction))

    return {
        "image": image_name,
        "predicted_class": class_names[predicted_index],
        "confidence": float(prediction[predicted_index]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    parser.add_argument("--data-dir", default="dataset/raw")
    parser.add_argument("--output-dir", default="submission")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=150)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset == "cifar10":
        dataset = load_cifar10(args.batch_size)
    else:
        dataset = load_image_folder(Path(args.data_dir), (args.image_size, args.image_size), args.batch_size)

    class_names = dataset["class_names"]
    image_size = dataset["image_size"]
    train_ds = dataset["train_ds"]
    val_ds = dataset["val_ds"]
    test_ds = dataset["test_ds"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(image_size, len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_history(history, output_dir)
    export_models(model, output_dir, image_size)

    metadata = {
        "dataset": dataset["dataset_name"],
        "total_images": int(dataset["total_images"]),
        "classes": class_names,
        "train_images": int(dataset["train_images"]),
        "validation_images": int(dataset["validation_images"]),
        "test_images": int(dataset["test_images"]),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }
    metadata["tflite_inference_example"] = run_inference_example(
        output_dir / "tflite" / "model.tflite",
        dataset["sample_image"],
        class_names,
        image_size,
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for file_name in ["README.md", "requirements.txt"]:
        source = Path(file_name)
        if source.exists():
            shutil.copy2(source, output_dir / file_name)


if __name__ == "__main__":
    main()
