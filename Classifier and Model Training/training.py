import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# configs
DATA_DIR = pathlib.Path("dataset")
IMG_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 50 
MODEL_NAME = "watch_esp32"
VAL_SPLIT = 0.2
SEED = 123
NUM_IMAGES = 24


def make_datasets():
    # Training subset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=True,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
    )

    # Validation subset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=False,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # grayscale + normalize [0,1]
    def preprocess(image, label):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = (
        train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(100)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Conv2D(
            16, 3, activation="relu",
            input_shape=(IMG_SIZE, IMG_SIZE, 1)
        ),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_dataset():
    rep_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        color_mode="rgb",
        shuffle=True,
    ).take(NUM_IMAGES)

    for img, _ in rep_ds:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img, tf.float32) / 255.0
        yield [img]


def quantize_to_int8(model: keras.Model, tflite_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to", tflite_path)


def write_c_array(tflite_path: str, header_path: str, array_name="g_model"):
    data = pathlib.Path(tflite_path).read_bytes()

    with open(header_path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"extern const unsigned char {array_name}[];\n")
        f.write(f"extern const unsigned int {array_name}_len;\n\n")
        f.write(f"const unsigned char {array_name}[] = {{\n")

        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")

        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")
    print("Wrote C array to", header_path)


def write_labels(class_names, path="labels.txt"):
    with open(path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Saved labels to", path, "->", class_names)


def main():
    train_ds, val_ds, class_names = make_datasets()
    write_labels(class_names)

    model = build_model(len(class_names))
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    model.save(f"{MODEL_NAME}_fp32.keras")

    tflite_path = f"{MODEL_NAME}_int8.tflite"
    quantize_to_int8(model, tflite_path)

    write_c_array(tflite_path, "model_data.h")


if __name__ == "__main__":
    main()