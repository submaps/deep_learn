from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimesion=10000):
    results = np.zeros((len(sequences), dimesion))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    word_index = imdb.get_word_index()
    reverse_words_index = dict([(v, k) for k, v in word_index.items()])
    decoded_review = ' '.join([reverse_words_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_review)

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid', input_shape=(10000,)))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train, partial_y_train, epochs=4,
                            batch_size=512,
                            validation_data=(x_val, y_val))
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test_acc:', test_acc)

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    plot_loss(loss_values, val_loss_values)

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plot_acc(acc_values, val_acc_values)


def plot_loss(loss_values, val_loss_values):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc(acc_values, val_acc_values):
    epochs = range(len(acc_values))
    plt.clf()
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
