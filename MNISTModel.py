from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasClassifier

class MNISTModel:
    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.model = self.create_model()
        
    def show_img(self, image_index):
        selected_image = self.train_images[image_index]
        plt.imshow(selected_image, cmap='gray')
        plt.title(f'Label: {self.train_labels[image_index]}')
        plt.show()
    
    def create_model(self, num_units=128, activation='relu', **kwargs):
        model = Sequential()
        model.add(Input(shape=(28, 28)))
        model.add(Flatten())
        model.add(Dense(num_units, activation=activation))
        model.add(Dense(num_units, activation=activation))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def tune_hyperparameters(self, param_grid):
        
        model_callable = lambda **kwargs: self.create_model(**kwargs)
        model = KerasClassifier(model=model_callable, verbose=0)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=3, n_jobs=-1)
        grid_result = grid.fit(self.train_images, self.train_labels)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result
    
    def get_model_summary(self):
        self.model.summary()

    def train_model(self, batch_size=32, epochs=1):
        history = self.model.fit(
            self.train_images, self.train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.test_images, self.test_labels)
        )
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()