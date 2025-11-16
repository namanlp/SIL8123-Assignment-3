from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.datasets import cifar10

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
N = 10000
NUM_CLASSES = 10

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------- DEFENSE SECTION -------------------

def predict_with_temperature(model, x, T=10.0):
    logits = model.predict(x, verbose=0)
    logits = logits / T
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
model.fit(*cifar10.load_data()[0], epochs=1, batch_size=256, verbose=0)

# ---------------------------------------------------------

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0

x_member  = x_train[:N]
x_nonmem  = x_test[:N]

p_member    = predict_with_temperature(model, x_member, T=10)
p_non_member = predict_with_temperature(model, x_nonmem, T=10)

s_member, s_nonmem = np.max(p_member,1), np.max(p_non_member,1)

labels = np.concatenate([np.ones(N), np.zeros(N)])
scores =  np.concatenate([s_member, s_nonmem])
threshold = np.mean(scores)

preds = (scores > threshold).astype(int)

print("Accuracy:",  accuracy_score(labels, preds))
print("Precision:", precision_score(labels, preds, zero_division=0))
print("Recall:",    recall_score(labels, preds, zero_division=0))
