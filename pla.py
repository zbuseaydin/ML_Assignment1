import numpy as np
import matplotlib.pyplot as plt

large_data = np.load('pla_data/data_large.npy')
small_data = np.load('pla_data/data_small.npy')
large_labels = np.load('pla_data/label_large.npy')
small_labels = np.load('pla_data/label_small.npy')

def perceptron(weight, data, labels, max_iter=10000):
    for iteration in range(max_iter):
        prev_weight = weight.copy()
        for i in np.random.permutation(len(data)):
            if np.dot(weight, data[i]) * labels[i] <= 0:
                weight += data[i] * labels[i]
                break
        if np.array_equal(prev_weight, weight):
            break
    return weight, iteration

def test(weight, data, labels):
    correct = 0
    for i in range(len(data)):
        if np.dot(weight, data[i]) * labels[i] > 0:
            correct += 1
    return correct / len(data)

def split_data(data, labels, train_split):
    train_size = int(train_split * len(data))
    train_set = data[:train_size]
    train_labels = labels[:train_size]
    test_set = data[train_size:]
    test_labels = labels[train_size:]
    return train_set, train_labels, test_set, test_labels

def plot(data, labels, weight, name):
    plt.scatter(data[labels == 1][:,1], data[labels == 1][:,2], marker='o', label='1')
    plt.scatter(data[labels == -1][:,1], data[labels == -1][:,2], marker='x', label='-1')
    x = np.linspace(0, 1, 100)
    # y = -ax/b - c/b
    y = -weight[1] / weight[2] * x - weight[0] / weight[2]
    plt.plot(x, y, label='Decision boundary', color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(name)
    plt.clf()


if __name__ == '__main__':
    small_data_iter = []
    large_data_iter = []
    accuracy_small_data = []
    accuracy_large_data = []
    stats = open('pla_results/stats.txt', 'w')
    large_train_set, large_train_labels, large_test_set, large_test_labels = split_data(large_data, large_labels, 0.8)
    small_train_set, small_train_labels, small_test_set, small_test_labels = split_data(small_data, small_labels, 0.8)
    for s in range(11):
        seed = s
        np.random.seed(seed)
        if s == 0:
            small_weight = np.zeros(len(small_train_set[0]))
            large_weight = np.zeros(len(large_train_set[0]))
            large_weight, covergence_iter_large = perceptron(large_weight, large_train_set, large_train_labels)
            stats.write("Small data set:\n")
            plot(large_train_set, large_train_labels, large_weight, "pla_results/large_train.png")
        else:
            small_weight = np.random.rand(len(small_train_set[0])) * 10
            large_weight = np.random.rand(len(large_train_set[0])) * 10
            stats.write("Seed: {}\n".format(seed))
        stats.write("Initial weights for small dataset: {}\n".format(np.array2string(small_weight, formatter={'float_kind':lambda x: "%.2f" % x})))
        small_weight, convergence_iter_small = perceptron(small_weight, small_train_set, small_train_labels)
        large_weight, covergence_iter_large = perceptron(large_weight, large_train_set, large_train_labels)
        small_data_iter.append(convergence_iter_small)
        large_data_iter.append(covergence_iter_large)
        accuracy_small = test(small_weight, small_test_set, small_test_labels)
        accuracy_large = test(large_weight, large_test_set, large_test_labels)
        accuracy_small_data.append(accuracy_small)
        accuracy_large_data.append(accuracy_large)
        stats.write("Convergence iteration: {}\n".format(convergence_iter_small))
        stats.write("Accuracy: {:.2f}\n".format(accuracy_small))
        stats.write("Weights: {}\n".format(np.array2string(small_weight, formatter={'float_kind':lambda x: "%.2f" % x})))
        stats.write("Decision boundary: y = {:.2f}x + {:.2f}\n".format(-small_weight[1] / small_weight[2], -small_weight[0] / small_weight[2]))
        stats.write("--------------------\n")
        
        plot(small_train_set, small_train_labels, small_weight, "pla_results/small_train_{}.png".format(seed))
    stats.write("Average number of iterations for convergence on the small dataset: {:.2f}\n".format(np.mean(small_data_iter)))
    stats.write("Average number of iterations for convergence on the large dataset: {:.2f}\n".format(np.mean(large_data_iter)))
    stats.write("Average accuracy on the small dataset: {}\n".format(np.mean(accuracy_small_data)))
    stats.write("Average accuracy on the large dataset: {}\n".format(np.mean(accuracy_large_data)))
    stats.close()

    
