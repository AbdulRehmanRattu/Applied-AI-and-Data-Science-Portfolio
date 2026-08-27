import numpy as np
import time
import matplotlib.pyplot as plt

def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = [int(line.strip()) if int(line.strip()) > 0 else 0 for line in lines]
    return labels, len(labels)

def load_images(file_path, num_images, pooling_factor):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    height = len(lines) // num_images
    width = len(lines[0])
    images = []
    for index in range(num_images):
        image = np.zeros((height, width))
        for row in range(height):
            line = lines[index * height + row]
            for col, char in enumerate(line):
                if char in ['+', '#']:
                    image[row, col] = 1
        images.append(image)

    # Pooling step to reduce image dimension
    new_height = height // pooling_factor
    new_width = width // pooling_factor
    pooled_images = np.zeros((num_images, new_height, new_width))
    for i in range(num_images):
        for new_row in range(new_height):
            for new_col in range(new_width):
                pooling_area = images[i][new_row*pooling_factor:(new_row+1)*pooling_factor, new_col*pooling_factor:(new_col+1)*pooling_factor]
                pooled_images[i, new_row, new_col] = np.sum(pooling_area)
    return pooled_images

def preprocess_data(image_path, label_path, pool_size):
    labels, num_samples = read_labels(label_path)
    images = load_images(image_path, num_samples, pool_size)
    flat_images = [image.flatten() for image in images]
    indices = np.arange(len(flat_images))
    np.random.shuffle(indices)
    return np.array(flat_images)[indices], np.array(labels)[indices]

def calculate_probabilities(features, labels, pool_size):
    num_labels = np.unique(labels).size
    num_features = features.shape[1]
    count_features = np.zeros((num_labels, num_features, pool_size*pool_size+1))
    label_count = np.zeros(num_labels)
    for feature_vector, label in zip(features, labels):
        label_count[label] += 1
        for idx, feature in enumerate(feature_vector):
            count_features[label, idx, int(feature)] += 1

    probability_features = count_features / label_count[:, None, None]
    prior_probabilities = label_count / sum(label_count)
    return probability_features, prior_probabilities

def classify(test_data, feature_probs, priors):
    num_labels = feature_probs.shape[0]
    probabilities = np.ones((num_labels, len(test_data)))
    for label in range(num_labels):
        for idx, data_point in enumerate(test_data):
            for feature_idx, feature in enumerate(data_point):
                feature_prob = max(feature_probs[label, feature_idx, int(feature)], 0.001)
                probabilities[label, idx] *= feature_prob
            probabilities[label, idx] *= priors[label]
    return np.argmax(probabilities, axis=0)

def calculate_accuracy(predictions, actuals):
    return np.mean(predictions == actuals)

def visualize_performance(data, title, color, label):
    percentages = np.linspace(0.1, 1.0, 10)
    plt.plot(percentages, data, label=label, color=color)
    plt.xlabel('Proportion of Training Data')
    plt.title(title)
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()

def run():
    train_data_path = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/facedata/facedatatrain"
    train_labels_path = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/facedata/facedatatrainlabels"
    test_data_path = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/facedata/facedatatest"
    test_labels_path = "C:/Users/Abdul Rehman/Downloads/Project Perfect/Classifiers/code/facedata/facedatatestlabels"

    pooling_size = 3
    train_features, train_labels = preprocess_data(train_data_path, train_labels_path, pooling_size)
    test_features, test_labels = preprocess_data(test_data_path, test_labels_path, pooling_size)
    segments = 10
    timing = []
    accuracies = []
    for i in range(segments):
        start_time = time.time()
        feature_probabilities, priors = calculate_probabilities(train_features[:int((i + 1) / segments * len(train_features))],
                                                                train_labels[:int((i + 1) / segments * len(train_labels))], pooling_size)
        predictions = classify(test_features, feature_probabilities, priors)
        accuracy = calculate_accuracy(predictions, test_labels)
        end_time = time.time()
        timing.append(end_time - start_time)
        accuracies.append(accuracy)
        print(f"Test accuracy: {accuracy}")
    visualize_performance(timing, 'Training Time', 'blue', 'Time (s)')
    visualize_performance(accuracies, 'Accuracy', 'red', 'Accuracy')

run()
