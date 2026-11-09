import pandas as pd
import tensorflow as tf

def load_manifest(csv_path, class_to_id, image_size, batch_size, shuffle=False, seed=42):
    table = pd.read_csv(csv_path)
    labels = table["label"].map(class_to_id).to_numpy(dtype="int32")
    dataset = tf.data.Dataset.from_tensor_slices((table["path"].to_numpy(), labels))
    if shuffle:
        dataset = dataset.shuffle(len(table), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: (
            tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), image_size) / 255.0,
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_augmentation(seed=42):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05, seed=seed),
        tf.keras.layers.RandomZoom(0.08, seed=seed),
        tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        tf.keras.layers.RandomContrast(0.10, seed=seed),
    ])
