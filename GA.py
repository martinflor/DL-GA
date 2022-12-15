import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import numpy as np

# Define the CNN architecture
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

# Load the Fashion-MNIST data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convert the labels to one-hot encoded format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshape the data to have a single channel
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# Convert the data to floating-point type and normalize it
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train[:1000]
y_train = y_train[:1000]
X_test = X_test[:1000]
y_test = y_test[:1000]

from numba import jit

def fitness(model):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Evaluate the model on the test data
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    # Return the model accuracy as the fitness value
    return acc,

@jit(forceobj=True)
def cx_cnn(ind1, ind2):
    """Custom crossover operator for CNNs. This operator swaps the weights of the
    individual layers between the parent models to produce the offspring.
    """
    # Create copies of the parent models
    off1 = ind1.__class__.from_config(ind1.get_config())
    off2 = ind2.__class__.from_config(ind2.get_config())
    
    # Swap the weights of the individual layers between the parent models
    for i, (layer1, layer2) in enumerate(zip(ind1.layers, ind2.layers)):
        weights1 = layer1.get_weights()
        weights2 = layer2.get_weights()
        weights1 = [np.array(w) for w in weights1]  # Convert weights to NumPy arrays
        weights2 = [np.array(w) for w in weights2]  # Convert weights to NumPy arrays
        off1.layers[i].set_weights(weights2)
        off2.layers[i].set_weights(weights1)
    
    return off1, off2

@jit(forceobj=True)
def mut_cnn(model):
    """Custom mutation operator for CNNs. This operator adds Gaussian noise to
    the weights of the individual layers of the model.
    """
    # Create a copy of the model
    off = model.__class__.from_config(model.get_config())
    
    # Add Gaussian noise to the weights of the individual layers
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        weights = [np.array(w) for w in weights]  # Convert weights to NumPy arrays
        weights = [w + np.random.normal(0, 0.1, size=w.shape) for w in weights]  # Add noise to weights
        off.layers[i].set_weights(weights)
    
    return off,

# Define the selection operator
def sel_tournament(population, k):
    """Tournament selection operator. This operator selects the best individual
    from a random sample of size k from the population.
    """
    # Select a random sample of size k from the population
    sample = np.random.choice(population, size=k, replace=False)
    # Return the best individual from the sample
    return max(sample, key=fitness)

# Define the GA parameters
POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 10

# Define the population
population = [create_model() for _ in range(POPULATION_SIZE)]


train_acc = []
test_acc = []


# Run the GA
for generation in range(MAX_GENERATIONS):
    print(generation)
    # Evaluate the fitness of the population
    fitnesses = [fitness(model) for model in population]
    
    # Select the best-performing individuals
    #population = [population[i] for i in np.argsort(fitnesses)[::-1]]
    
    # Crossover
    off1 = []
    off2 = []
    for i in range(0, len(population) - 1, 2):
        if np.random.random() < P_CROSSOVER:
            tmp = cx_cnn(population[i], population[i + 1])
            off1.append(tmp[0])
            off2.append(tmp[1])
        else:
            off1.append(population[i])
            off2.append(population[i + 1])
    population = off1 + off2
    
    # Mutation
    for i in range(len(population)):
        if np.random.random() < P_MUTATION:
            population[i] = mut_cnn(population[i])[0]
            
    # Selection
    population = [sel_tournament(population, k=3) for _ in population]

    _, train_accuracy = population[0].evaluate(X_train, y_train, verbose=0)
    _, test_accuracy = population[0].evaluate(X_test, y_test, verbose=0)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    
    
    
# Print the best individual and its fitness
best_individual = population[0]
best_fitness = fitness(best_individual)[0]
print("Best individual: ", best_individual)
print("Best fitness: ", best_fitness)

import matplotlib.pyplot as plt

plt.plot(train_acc, label='Training accuracy')
plt.plot(test_acc, label='Test accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

       
