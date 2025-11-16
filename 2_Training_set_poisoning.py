from os import environ, makedirs
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
NUM_CLASSES = 10
POISON_FRAC = 0.20
CLEAN_FRAC_TO_KEEP = 0.90   # smallest possible defense

model = tf.keras.models.load_model(MODEL_PATH)

(x_train, y_train_raw), (x_test, y_test_raw) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train_raw, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test_raw, NUM_CLASSES)

# --------------------------
# (a) SIMPLE LABEL-FLIP POISONING
# --------------------------
n = int(len(y_train) * POISON_FRAC)
idx = np.random.choice(len(y_train), n, replace=False)
y_poison = y_train.copy()

for i in idx:
    true = np.argmax(y_train[i])
    choices = [c for c in range(NUM_CLASSES) if c != true]
    y_poison[i] = np.eye(NUM_CLASSES)[np.random.choice(choices)]

# --------------------------
# (b) DEFENSE: LOSS-BASED SANITIZATION (SMALLEST POSSIBLE)
# --------------------------
preds = model.predict(x_train, verbose=0)
losses = tf.keras.losses.categorical_crossentropy(y_poison, preds).numpy()

th = np.percentile(losses, CLEAN_FRAC_TO_KEEP * 100)   # keep 90% smallest
keep_mask = losses <= th

x_clean = x_train[keep_mask]
y_clean = y_poison[keep_mask]

print("Kept clean samples:", len(x_clean), "/", len(x_train))

# --------------------------
# (c) RETRAIN ON CLEANED DATASET
# --------------------------
model.fit(x_clean, y_clean, epochs=10, batch_size=64, verbose=1)

# --------------------------
# (d) EVALUATE
# --------------------------
pred = model.predict(x_test, verbose=0)
from numpy import argmax
print("Clean test accuracy:", np.mean(argmax(pred,1)==argmax(y_test,1)))

# --------------------------
# (e) SAVE SAMPLE POISONED IMAGES
# --------------------------
SAVE_DIR = "poison_defense_results"
makedirs(SAVE_DIR, exist_ok=True)

for i in idx[:20]:  # save few examples
    img = (x_train[i] * 255).astype("uint8")
    o = int(np.argmax(y_train[i]))
    p = int(np.argmax(y_poison[i]))
    Image.fromarray(img).save(f"{SAVE_DIR}/idx{i}_orig{o}_flip{p}.png")