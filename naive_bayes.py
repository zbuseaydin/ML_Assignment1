import math

# calculate sample mean
def mean(samples):
    sum = 0
    for sample in samples:
        sum += float(sample)
    return sum/len(samples)


# calculate the variance of the samples
def variance(samples):
    variance = 0
    sample_mean = mean(samples)
    for sample in samples:
        variance += (float(sample)-sample_mean) ** 2
    return variance/(len(samples)-1)


# calculate the prior probabilities of 
# diagnosis M and diagnosis B
def prior_probabilities(data_dict):
    num_m = len(data_dict["M"][0])
    prior_m = num_m / (len(data_dict["B"][0]) + num_m)
    return {"M": prior_m, "B": 1-prior_m}


# calculate the normal probability density function
def normal_pdf(mean, variance, x):
    prob = 1/(2 * math.pi * variance) ** 0.5
    exp = math.exp((-1 * (x-mean) ** 2) / (2 * variance))
    return prob * exp


def train(data):
    training_set = {"M": {}, "B": {}}
    # {
    #  "M": {0: [values of 0th attribute], 1: [values of 1st attribute], ...},
    #  "B": {0: [values of 0th attribute], 1: [values of 1st attribute], ...}
    # } 

    num_attributes = len(data[0])-2
    for instance in data:     # instances
        diagnosis = instance[1]

        for j in range(num_attributes):   # attributes
            if training_set[diagnosis].get(j) == None:
                training_set[diagnosis][j] = []
            training_set[diagnosis][j].append(instance[j+2])

    prior_probs = prior_probabilities(training_set)
    stats = {"M": {}, "B": {}}
    attributes_given_m = training_set["M"]
    attributes_given_b = training_set["B"]

    # calculate the mean and variances of each attribute given diagnosis
    for attr in range(num_attributes):
        stats["M"][attr] = [mean(attributes_given_m[attr]), variance(attributes_given_m[attr])]
        stats["B"][attr] = [mean(attributes_given_b[attr]), variance(attributes_given_b[attr])]

    # get the number of misclassifications from the training data using trained classifier
    train_misclassified = test(stats, prior_probs, training_data)
    return stats, prior_probs, train_misclassified


def classify(input, stats, prior_probs):
    probability_m = prior_probs["M"]
    # probability of M is proportional to the product of pdf values of the attributes given M
    for i in range(2, len(input)):
        probability_m *= normal_pdf(stats["M"][i-2][0], stats["M"][i-2][1], float(input[i]))
    
    probability_b = prior_probs["B"]
    # probability of B is proportional to the product of pdf values of the attributes given B
    for i in range(2, len(input)):
        probability_b *= normal_pdf(stats["B"][i-2][0], stats["B"][i-2][1], float(input[i]))

    # decide B if probability_b is larger
    if probability_b > probability_m:
        return "B"
    return "M"


def test(stats, prior_probs, data):
    misclassified = 0
    for instance in data:
        decision = classify(instance, stats, prior_probs)
        if decision != instance[1]:
            misclassified += 1
    return misclassified


if __name__ == "__main__":
    data_file = open("./breast+cancer+wisconsin+diagnostic/wdbc.data")
    data = data_file.read().split("\n")[:-1]
    training_size = int(len(data) * 0.8)

    training_data = [data[i].split(",") for i in range(training_size)]
    test_data = [data[i].split(",") for i in range(training_size, len(data))]
    
    stats, prior_probs, train_misclassified = train(training_data)
    test_misclassified = test(stats, prior_probs, test_data)
    train_accuracy = (1 - train_misclassified/training_size) * 100
    test_accuracy = (1 - test_misclassified/len(test_data)) * 100
    print(f"Training accuracy: {train_accuracy:.3f}%")
    print(f"Test accuracy: {test_accuracy:.3f}%")