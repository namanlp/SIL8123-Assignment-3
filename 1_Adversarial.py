# Adversarial Training (PGD)
import tensorflow as tf
eps, alpha, steps = 8/255, 2/255, 5

def pgd(model, x, y):
    x_adv = tf.identity(x)
    for _ in range(steps):
        with tf.GradientTape() as t:
            t.watch(x_adv)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, model(x_adv))
        g = tf.sign(t.gradient(loss, x_adv))
        x_adv = tf.clip_by_value(x_adv + alpha*g, x-eps, x+eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv

for epoch in range(5):
    for xb, yb in train_ds:
        xb_adv = pgd(model, xb, yb)
        model.train_on_batch(xb_adv, yb)