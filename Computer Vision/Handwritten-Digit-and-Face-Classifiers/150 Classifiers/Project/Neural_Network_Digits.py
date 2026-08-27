import numpy as np
import time
import matplotlib.pyplot as plt

def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = [int(line.strip()) for line in lines]
    return labels, len(labels)

def read_images(file_path, num_images):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    height = len(lines) // num_images
    width = len(lines[0])
    images = []
    for index in range(num_images):
        image = np.zeros((height, width))
        row_counter = 0
        for row in range(height * index, height * (index + 1)):
            line = lines[row]
            for col, char in enumerate(line):
                if char in ["+", "#"]:
                    image[row_counter, col] = 1
            row_counter += 1
        images.append(image)
    return images

def convert_to_one_hot(labels):
    one_hot_labels = []
    for label in labels:
        one_hot_vector = [0] * 10
        one_hot_vector[label] = 1
        one_hot_labels.append(one_hot_vector)
    return np.array(one_hot_labels)

def preprocess_data(image_file, label_file):
    labels, num_images = read_labels(label_file)
    images = read_images(image_file, num_images)
    labels = convert_to_one_hot(labels)
    flattened_images = [image.flatten() for image in images]
    indices = np.random.permutation(len(flattened_images))
    return np.array(flattened_images)[indices], labels[indices]

def update_parameters(weights, biases, x_train, y_train, iterations, learning_rate):
    for _ in range(iterations):
        grad_w, grad_b, cost = compute_gradients(weights, biases, x_train, y_train)
        weights -= learning_rate * grad_w
        biases -= learning_rate * grad_b
    return weights, biases

def compute_gradients(weights, biases, x_train, y_train):
    m = x_train.shape[0]
    activations = sigmoid(np.dot(x_train, weights) + biases)
    cost = -(1 / m) * np.sum(y_train * np.log(activations) + (1 - y_train) * np.log(1 - activations))
    grad_w = (1 / m) * np.dot(x_train.T, (activations - y_train))
    grad_b = (1 / m) * np.sum(activations - y_train)
    return grad_w, grad_b, cost

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_classes(weights, biases, x):
    probabilities = sigmoid(np.dot(x, weights) + biases)
    predictions = np.array([np.eye(10)[np.argmax(row)] for row in probabilities])
    return predictions

def calculate_accuracy(predictions, actual):
    return np.mean(np.all(predictions == actual, axis=1))

def visualize_data(metrics, title, color, y_label):
    x_axis = np.linspace(0.1, 1.0, 10)
    plt.plot(x_axis, metrics, label='Performance', color=color)
    plt.xlabel('Fraction of Training Data Used')
    plt.title(title)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def execute():
    training_images = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/digitdata/trainingimages"
    training_labels = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/digitdata/traininglabels"
    testing_images = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/digitdata/testimages"
    testing_labels = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/digitdata/testlabels"

    x_train, y_train = preprocess_data(training_images, training_labels)
    x_test, y_test = preprocess_data(testing_images, testing_labels)
    training_steps = x_train.shape[0] // 10
    timings = []
    accuracies = []
    for i in range(10):
        start_time = time.time()
        weights = np.zeros((x_train.shape[1], 10))
        biases = np.zeros(10)
        weights, biases = update_parameters(weights, biases, x_train[:training_steps * (i + 1)], y_train[:training_steps * (i + 1)], 2000, 0.6)
        end_time = time.time()
        predictions = predict_classes(weights, biases, x_test)
        accuracy = calculate_accuracy(predictions, y_test)
        timings.append(end_time - start_time)
        accuracies.append(accuracy)
        print(f"Test accuracy: {accuracy}")
    visualize_data(timings, 'Digit Image Training Time', 'blue', 'Time (s)')
    visualize_data(accuracies, 'Digit Image Accuracy', 'red', 'Accuracy')

execute()
