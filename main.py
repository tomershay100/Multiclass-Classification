import sys
import math
import numpy as np


# calculate distance between two points
def dist_two_points(p1, p2):
    temp = 0
    for i in range(len(p1)):
        temp += (p1[i] - p2[i]) ** 2
    return math.sqrt(temp)


def knn(train_data, train_labels, test_data, k=1):
    test_labels = []

    for i in range(len(test_data)):

        k_neighbors_label = []
        k_neighbors_dist = []
        for _ in range(k):
            k_neighbors_dist.append(-1)
            k_neighbors_label.append(0)

        test_point = test_data[i]

        # for all test point, go over all train points
        for train_point, train_label in zip(train_data, train_labels):
            # calculate distance from test point to train point
            curr_dist = dist_two_points(test_point, train_point)

            # If 5 iterations haven't passed, add the train point to the list of the closest k
            if -1 in k_neighbors_dist:
                for e in range(len(k_neighbors_dist)):
                    if k_neighbors_dist[e] == -1:
                        k_neighbors_dist[e] = curr_dist
                        k_neighbors_label[e] = train_label
                        break
            else:
                max_i = np.argmax(k_neighbors_dist)
                # If the current point is closer than all the points in the nearest k list
                if curr_dist < k_neighbors_dist[max_i]:
                    # Replace the current point with the farthest point in the list
                    k_neighbors_dist[max_i] = curr_dist
                    k_neighbors_label[max_i] = train_label
        # Set the test point to be like most of the k points closest to it
        test_labels.append(np.bincount(k_neighbors_label).argmax())
    return test_labels


# split train data into training and validation set
def to_validate(train_data, train_labels, percent=10):
    # randomized all train data set
    randomize = np.arange(len(train_data))
    np.random.shuffle(randomize)
    train_data = train_data[randomize]
    train_labels = train_labels[randomize]

    from_i = float(0 * len(train_data)) / percent
    to_i = float((0 + 1) * len(train_data)) / percent

    validate_data = train_data[int(from_i): int(to_i)]
    validate_labels = train_labels[int(from_i): int(to_i)]

    new_train_data = train_data[0: int(from_i)]
    if len(new_train_data) == 0:
        new_train_data = train_data[int(to_i):]
    else:
        new_train_data = np.append(new_train_data, train_data[int(to_i):], axis=0)

    new_train_labels = train_labels[0: int(from_i)]
    if len(new_train_labels) == 0:
        new_train_labels = train_labels[int(to_i):]
    else:
        new_train_labels = np.append(new_train_labels, train_labels[int(to_i):])

    return validate_data, validate_labels, new_train_data, new_train_labels


# normalize x row by z score normalization
def norm_row_z_score(x, stds, means):
    new_x = []
    for i in range(len(x)):
        new_x.append((x[i] - means[i]) / stds[i])
    return np.array(new_x)


# calculate std and mean for all feature in train data
def z_score_normalize(train_data):
    transpose = np.transpose(train_data)
    stds = []
    means = []
    for feature in transpose:
        stds.append(np.std(feature))
        means.append(np.mean(feature))
    return stds, means


# returns prediction of x row with w weights
def predict(x, w):
    predictions = [np.dot(w[0], x), np.dot(w[1], x), np.dot(w[2], x)]
    predictions = np.array(predictions)
    y_hat = np.argmax(predictions)
    return y_hat


# returns the count of the right prediction
def make_prediction(data, labels, w):
    count = 0
    for x, y in zip(data, labels):
        # add feature for bias
        x = np.append(x, 1)

        # get prediction
        y_hat = predict(x, w)
        y = int(y)
        if y_hat == y:
            count += 1
    return count


# returns list of all predictions
def test_prediction(best_w, test_data):
    test_labels = []
    for x in test_data:
        # add feature for bias
        x = np.append(x, 1)

        # get prediction
        y_hat = predict(x, best_w)

        test_labels.append(y_hat)
    return test_labels


# init the weights, lr, normalization and validate
def init_algorithms(train_data, train_labels, validate_part=3):
    # weights init to zeros
    w = [np.zeros(len(train_data[0]) + 1), np.zeros(len(train_data[0]) + 1), np.zeros(len(train_data[0]) + 1)]
    w = np.array(w)

    # eta
    lr = 0.001

    # std and mean for all feature - for the normalization
    stds, means = z_score_normalize(train_data)

    # 1/validate_part to validation set
    validate_data, validate_labels, new_train_data, new_train_labels = to_validate(train_data, train_labels,
                                                                                   percent=validate_part)
    return lr, means.copy(), stds.copy(), validate_data.copy(), validate_labels.copy(), w.copy()


def perceptron(train_data, train_labels, test_data, validate_data, validate_labels, w, lr):
    # best weights
    best_w = []
    best_validate = 0

    for epoch in range(1, 50):
        # every 10 epoch, eta divided by 10
        if epoch % 10 == 0:
            lr *= 0.1

        # shuffle train data set
        randomize = np.arange(len(train_data))
        np.random.shuffle(randomize)
        train_data = train_data[randomize]
        train_labels = train_labels[randomize]

        for x, y in zip(train_data, train_labels):
            # add feature for bias
            x = np.append(x, 1)

            # get prediction
            y_hat = predict(x, w)

            y = int(y)
            if y_hat != y:
                # update model
                w[y_hat] = w[y_hat] - lr * x
                w[y] = w[y] + lr * x

        # check on validate
        count = make_prediction(validate_data, validate_labels, w)
        acc = 100 * count / len(validate_data)
        if acc > best_validate:
            best_validate = acc
            best_w = w.copy()

    # make prediction on test
    test_labels = test_prediction(best_w, test_data)
    return test_labels


def loss_svm(w_y, x, w_r):
    tmp = 1 - np.dot(w_y, x) + np.dot(w_r, x)
    return max(0, tmp)


# returns prediction of x row with w weights for svm (y_hat != y)
def predict_svm(x, w, y):
    predictions = [np.dot(w[0], x), np.dot(w[1], x), np.dot(w[2], x)]
    predictions = np.array(predictions)
    predictions[y] = float('-inf')
    y_hat = np.argmax(predictions)
    return y_hat


def svm(train_data, train_labels, test_data, validate_data, validate_labels, w):
    # best weights
    best_w = []
    best_validate = 0

    # eta and lambda
    eta_lr = 0.1
    lambda_lr = 0.001

    for epoch in range(1, 50):
        # every 5 epochs, eta divided by 2
        if epoch % 5 == 0:
            eta_lr *= 0.5
            # lambda_lr *= 2

        # shuffle train data set
        randomize = np.arange(len(train_data))
        np.random.shuffle(randomize)
        train_data = train_data[randomize]
        train_labels = train_labels[randomize]

        for x, y in zip(train_data, train_labels):
            # add feature for bias
            x = np.append(x, 1)

            y = int(y)

            # get prediction
            r = predict_svm(x, w, y)
            # assert y != r
            if loss_svm(w[y], x, w[r]) > 0:
                # update model
                w[r] = w[r] * (1 - eta_lr * lambda_lr) - eta_lr * x
                w[y] = w[y] * (1 - eta_lr * lambda_lr) + eta_lr * x

                tmp = np.array([0, 1, 2])
                tmp = np.delete(tmp, [r, y])
                s = tmp[0]

                w[s] = w[s] * (1 - eta_lr * lambda_lr)
            else:
                w[0] = w[0] * (1 - eta_lr * lambda_lr)
                w[1] = w[1] * (1 - eta_lr * lambda_lr)
                w[2] = w[2] * (1 - eta_lr * lambda_lr)

        # check on validate
        count = make_prediction(validate_data, validate_labels, w)
        acc = 100 * count / len(validate_data)
        if acc >= best_validate:
            best_validate = acc
            best_w = w.copy()

    # make prediction on test
    test_labels = test_prediction(best_w, test_data)
    return test_labels


# calculate tau of the PA algorithm
def tau_pa(w_y, x, w_r):
    tmp = 1 - np.dot(w_y, x) + np.dot(w_r, x)
    tmp = max(0, tmp)
    return tmp / (2 * np.dot(x, x) ** 2)


def pa(train_data, train_labels, test_data, validate_data, validate_labels, w):
    # best weights
    best_w = []
    best_validate = 0

    for epoch in range(1, 100):
        # shuffle train data set
        randomize = np.arange(len(train_data))
        np.random.shuffle(randomize)
        train_data = train_data[randomize]
        train_labels = train_labels[randomize]

        for x, y in zip(train_data, train_labels):
            # add feature for bias
            x = np.append(x, 1)

            # get prediction
            y_hat = predict(x, w)

            y = int(y)
            # tau calculation
            tau = tau_pa(w[y], x, w[y_hat])
            if y_hat != y:
                # update model
                w[y_hat] = w[y_hat] - tau * x
                w[y] = w[y] + tau * x

        # check on validate
        count = make_prediction(validate_data, validate_labels, w)
        acc = 100 * count / len(validate_data)
        if acc > best_validate:
            best_validate = acc
            best_w = w.copy()

    # make prediction on test
    test_labels = test_prediction(best_w, test_data)
    return test_labels


if len(sys.argv) < 5:
    print("not enough arguments!")
    exit(-1)

# get arguments
train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
output_log_name = sys.argv[4]

# get data from the files
train_x = np.loadtxt(train_x_path, dtype=float, delimiter=',')
train_y = np.loadtxt(train_y_path, dtype=float, delimiter=',')
test_x = np.loadtxt(test_x_path, dtype=float, delimiter=',')

# delete feature 3 (index 2)
train_x = np.transpose(train_x)
train_x = np.delete(train_x, 2, 0)
train_x = np.transpose(train_x)

test_x = np.transpose(test_x)
test_x = np.delete(test_x, 2, 0)
test_x = np.transpose(test_x)

# KNN - K Nearest Neighbors
knn_labels = knn(train_x, train_y, test_x, k=5)

# init the validate, normalize data and weights
learning_rate, means_train, stds_train, validate_x, validate_y, weights = init_algorithms(train_x, train_y,
                                                                                          validate_part=2)

# normalize by z-score normalization
for index in range(len(train_x)):
    train_x[index] = norm_row_z_score(train_x[index], stds_train, means_train)

for index in range(len(validate_x)):
    validate_x[index] = norm_row_z_score(validate_x[index], stds_train, means_train)

for index in range(len(test_x)):
    test_x[index] = norm_row_z_score(test_x[index], stds_train, means_train)

# Perceptron
perceptron_labels = perceptron(train_x, train_y, test_x, validate_x, validate_y, weights, learning_rate)

# SVM - Support Vector Machine
svm_labels = svm(train_x, train_y, test_x, validate_x, validate_y, weights)

# PA - Passive Aggressive
pa_labels = pa(train_x, train_y, test_x, validate_x, validate_y, weights)

# write prediction on the test to output_log file
f = open(output_log_name, 'w')
for knn_label, perceptron_label, svm_label, pa_label in zip(knn_labels, perceptron_labels, svm_labels, pa_labels):
    f.write(f"knn: {knn_label}, perceptron: {perceptron_label}, svm: {svm_label}, pa: {pa_label}\n")
f.close()
