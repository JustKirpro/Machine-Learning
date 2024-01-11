from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_training(training_history):
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def create_model():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(64, 64, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(rate=0.5))
    model.add(Dense(36, activation='softmax'))
    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()

    train_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/data/characters/preprocessed train'
    classes = list(string.digits + string.ascii_uppercase)
    image_size = (64, 64)

    data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_batches = data_generator.flow_from_directory(
        directory=train_path,
        classes=classes,
        batch_size=32,
        target_size=image_size,
        color_mode='grayscale',
        subset='training')
    valid_batches = data_generator.flow_from_directory(
        directory=train_path,
        classes=classes,
        batch_size=32,
        target_size=image_size,
        color_mode='grayscale',
        subset='validation')

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        filepath='/Users/kirpro/Kirpro/8 триместр/Распознавание образов/models/CNN(5)/saved-model-{epoch:02d}-{val_loss:.2f}.h5',
        verbose=1,
        monitor='val_loss')

    history = model.fit(
        x=train_batches,
        validation_data=valid_batches,
        epochs=100,
        verbose=2,
        callbacks=[checkpoint_callback])

    plot_training(history)

    test_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/data/characters/preprocessed test'
    test_batches = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        directory=test_path,
        classes=classes,
        batch_size=32,
        target_size=image_size,
        color_mode='grayscale',
        shuffle=False)

    predictions = model.predict(x=test_batches, verbose=0)
    print(f'Test accuracy = {round(accuracy_score(test_batches.classes, np.argmax(predictions, axis=-1)) * 100, 2)}%')

    models_path = '/Users/kirpro/Kirpro/8 триместр/Распознавание образов/models/CNN(5)'
    test_accuracies = dict()
    for model_name in os.listdir(models_path):
        if not model_name.startswith('.'):
            model = load_model(os.path.join(models_path, model_name))
            predictions = model.predict(x=test_batches, verbose=0)
            test_accuracies[model_name] = round(
                accuracy_score(test_batches.classes, np.argmax(predictions, axis=-1)) * 100, 2)

    sorted(test_accuracies, key=test_accuracies.get, reverse=True)

    model = load_model('/Users/kirpro/Kirpro/8 триместр/Распознавание образов/models/CNN(5)/saved-model-98-0.59.h5')
    predictions = model.predict(x=test_batches, verbose=0)
    print(f'Test accuracy = {round(accuracy_score(test_batches.classes, np.argmax(predictions, axis=-1)) * 100, 2)}%')