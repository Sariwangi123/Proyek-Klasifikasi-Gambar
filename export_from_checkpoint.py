from train_local_gpu import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    SUBMISSION_DIR,
    load_dataset,
    make_dataset,
    plot_history,
    run_tflite_inference,
)

import json
import os
import shutil
import subprocess
import sys
import tensorflow as tf


def build_inference_model(model):
    random_layers = (
        tf.keras.layers.RandomFlip,
        tf.keras.layers.RandomTranslation,
        tf.keras.layers.RandomZoom,
        tf.keras.layers.RandomRotation,
        tf.keras.layers.RandomContrast,
    )
    inference_layers = [layer for layer in model.layers if not isinstance(layer, random_layers)]
    inference_model = tf.keras.Sequential(inference_layers, name="cifar10_inference_model")
    inference_model.build((None, 32, 32, 3))
    inference_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return inference_model


def export_inference_models(model):
    saved_model_dir = SUBMISSION_DIR / "saved_model"
    tflite_dir = SUBMISSION_DIR / "tflite"
    tfjs_dir = SUBMISSION_DIR / "tfjs_model"
    for path in [saved_model_dir, tflite_dir, tfjs_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    model.export(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (tflite_dir / "model.tflite").write_bytes(tflite_model)
    (tflite_dir / "label.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")

    keras_export_path = SUBMISSION_DIR / "inference_model.h5"
    model.save(keras_export_path)

    stub_dir = SUBMISSION_DIR / "_tfjs_stubs"
    tfdf_stub = stub_dir / "tensorflow_decision_forests"
    tfdf_stub.mkdir(parents=True, exist_ok=True)
    (tfdf_stub / "__init__.py").write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stub_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorflowjs.converters.converter",
            "--input_format=keras",
            str(keras_export_path),
            str(tfjs_dir),
        ],
        check=True,
        env=env,
    )
    shutil.rmtree(stub_dir, ignore_errors=True)


def main():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {BEST_MODEL_PATH}")

    x_train_full, y_train_full, x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()
    train_ds = make_dataset(x_train, y_train)
    test_ds = make_dataset(x_test, y_test)

    model = build_inference_model(tf.keras.models.load_model(BEST_MODEL_PATH))
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    log_path = SUBMISSION_DIR / "training_log.csv"
    if log_path.exists():
        import csv

        with log_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                history["accuracy"].append(float(row["accuracy"]))
                history["val_accuracy"].append(float(row["val_accuracy"]))
                history["loss"].append(float(row["loss"]))
                history["val_loss"].append(float(row["val_loss"]))
        plot_history(history)

    export_inference_models(model)
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
