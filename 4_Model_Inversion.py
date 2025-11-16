# === DEFENSE 1: Label smoothing finetune (reduces overconfident logits) ===
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)
model.fit(x_train[:5000]/255.0,
          tf.keras.utils.to_categorical(y_train[:5000],10),
          epochs=1, batch_size=256, verbose=0)

# === DEFENSE 2: Temperature scaling (reduces inversion quality) ===
def predict_temp(model, x, T=20):              # HIGH temperature = LOW confidence
    logits = model(x, training=False).numpy()
    logits = logits / T
    e = np.exp(logits - np.max(logits,axis=1,keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# === OPTIONAL: Add small inference noise (1-line defense) ===
def noisy_logits(x):
    return x + np.random.normal(0, 0.02, x.shape)