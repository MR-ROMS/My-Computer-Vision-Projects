# train_cnn.py
# Trains the CNN model for traffic light detection.
# Run: python train_cnn.py

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 25
NUM_CLASS  = 5
MODEL_PATH = 'traffic_model.keras'

# FIX: Class names must match your dataset folder names exactly
# These are also used in traffic_detection.py — must be identical
CLASS_NAMES = ['Background', 'Green_Light', 'Left_Turn', 'Red_Light', 'Right_Turn']

# ── LOAD & PREPARE DATASET ──────────────────
def prepare_data():
    print("\n[*] Loading dataset...")
    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join('dataset', class_name)
        if not os.path.exists(class_path):
            print(f"  [WARN] Folder not found: {class_path}")
            continue

        count = 0
        for file in os.listdir(class_path):
            if file.lower().endswith(('.ppm', '.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_path, file)
                img = cv.imread(img_path)
                if img is None:
                    continue
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_idx)
                count += 1
        print(f"  [{class_idx}] {class_name}: {count} samples")

    if not images:
        print("ERROR: No images loaded — check your dataset folder path")
        return None, None, None

    X = np.array(images, dtype='float32')  # rescaling done inside model
    y = np.array(labels)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # Class weights — prevents model ignoring rare classes
    class_counts = np.bincount(y_train, minlength=NUM_CLASS)
    total = len(y_train)
    class_weight = {
        i: (total / (NUM_CLASS * count)) if count > 0 else 1.0
        for i, count in enumerate(class_counts)
    }
    print("\n  Class weights:", {k: round(v, 3) for k, v in class_weight.items()})

    # Warn about empty classes
    empty = [CLASS_NAMES[i] for i, c in enumerate(class_counts) if c == 0]
    if empty:
        print(f"\n  [!] WARNING: These classes have 0 samples: {empty}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val,   y_val  )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weight

# ── BUILD MODEL ──────────────────────────────
def build_model():
    # Augmentation — only applied during training automatically
    augmentation = models.Sequential([
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(factor=0.3),   # fog/dim simulation
        tf.keras.layers.RandomContrast(0.3),
        # NO horizontal flip — Left_Turn would become Right_Turn
        # NO vertical flip — lights are never upside down
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Rescale once — detection script passes raw float32, NOT /255
    x = layers.Rescaling(1./255)(inputs)
    x = augmentation(x)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASS, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── MAIN ─────────────────────────────────────
if __name__ == '__main__':
    train_ds, val_ds, class_weight = prepare_data()
    if train_ds is None:
        exit()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks
    )

    print(f"\nBest model saved to {MODEL_PATH}")

    # Save class mapping so you never lose track
    mapping_path = MODEL_PATH.replace('.keras', '_classes.txt')
    with open(mapping_path, 'w') as f:
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i} {name}\n")
    print(f"Class mapping saved to {mapping_path}")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history.history['loss'],     label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")