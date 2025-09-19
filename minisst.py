import tensorflow as tf
import matplotlib.pyplot as plt

# Загрузка датасета MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Размеры данных
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# Визуализация первых 5 изображений
for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Цифра: {y_train[i]}")
    plt.axis('off')
    plt.show()
