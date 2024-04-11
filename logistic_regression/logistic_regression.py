import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import math




def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#********************** Deliverable 1 **********************

def shuffle_data(X, y):
    data = pd.concat([X, y], axis=1)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def standardization(features):
    features = features.astype(float)
    for feature in features.columns:
        features.loc[:,feature] = (features.loc[:,feature] - np.mean(features.loc[:,feature])) / np.std(features.loc[:,feature])
    return features


def preprocess(X, y):
    X, y = shuffle_data(X, y)
    X  = standardization(X)
    return X, y

#**************************************************************
# split data  for 5 fold cross validation given the index of test set

def split_data(features, targets, test_fold = 0, n_folds=5):
    test_percent = 1 / n_folds
    test_size = int(features.shape[0] * test_percent)
    train_size = int(features.shape[0] - test_size)
    
    test_size += features.shape[0] - train_size - test_size  # if there is a remainder, add it to test_size

    [train_features, test_features, train_targets, test_targets] = [None] * 4
    for i in range(n_folds):
        if i == test_fold:
            test_features = features[i*test_size:(i+1)*test_size]
            test_targets = targets[i*test_size:(i+1)*test_size]
        else:
            if train_features is None :
                train_features = features[i*test_size:(i+1)*test_size]
                train_targets = targets[i*test_size:(i+1)*test_size]
            else:
                train_features = pd.concat([train_features, features[i*test_size:(i+1)*test_size]])
                train_targets = pd.concat([train_targets, targets[i*test_size:(i+1)*test_size]])

    return train_features,  test_features, train_targets,  test_targets


#********************** Deliverable 2 **********************
# Use 5-fold cross-validation to determine the value of the regularization parameter.

def determine_best_lambda(X, y):
    lambda_ = run_cross_validation(X, y, lambdas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
    print("Best Lambda: ", lambda_)
    return lambda_

def run_cross_validation(X, y, lambdas):
    X, y = preprocess(X, y)
    scores = cross_validation_score(X, y, lambdas, n_folds=5)
    best_lambda = lambdas[np.argmax(scores)]
    return best_lambda



def cross_validation_score(X, y, lambdas, n_folds=5):
    accuracies = []
    
    for lambda_ in lambdas:
        accuracies_for_lambda = []
        for i in range(n_folds):
            train_X, test_X, train_y, test_y = split_data(X, y, i, n_folds)
            weights, best_weights, _, _, _= logistic_regression(train_X, train_y, step_size=0.01, epochs=2000, stochastic=False, regularization=True, lambda_=lambda_)
            accuracy = test(test_X, test_y, weights) #  best weight secersem ne olur
            accuracies_for_lambda.append(accuracy)

        
        accuracies.append(accuracies_for_lambda)

    lambda_df = pd.DataFrame({'Lambda': lambdas, 'Accuracy average': [np.mean(x) for x in accuracies],  })
    for i, accuracy in enumerate(accuracies):
        for j, acc in enumerate(accuracy):
            lambda_df.loc[i, 'Fold ' + str(j + 1)] = acc

    lambda_df.to_csv('lambda.csv', index=False)
    return accuracies


#**************************************************************
# Use 5-fold cross-validation to determine the value of the step size parameter.

def determine_best_step_size(X, y):
    X, y = preprocess(X, y)
    step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
    accuracies, convergence_epochs = cross_validation_score_step_size(X, y, step_sizes, n_folds=5)




def cross_validation_score_step_size(X, y, step_sizes, n_folds=5):
    accuracies = []
    convergence_epochs = []
    
    for step_size in step_sizes:
        accuracies_for_step_size = []
        convergence_epochs_for_step_size = []
        for i in range(n_folds):
            train_X, test_X, train_y, test_y = split_data(X, y, i, n_folds)
            weights, best_weights, _, _, converged_epoch = logistic_regression(train_X, train_y, step_size=step_size, epochs=2500, stochastic=False, regularization=True, lambda_=0.05)
            accuracy = test(test_X, test_y, weights) 
            accuracies_for_step_size.append(accuracy)
            convergence_epochs_for_step_size.append(converged_epoch)

        
        accuracies.append(accuracies_for_step_size)
        convergence_epochs.append(convergence_epochs_for_step_size)

    step_size_df = pd.DataFrame({'Step Size': step_sizes, 'Accuracy average': [np.mean(x) for x in accuracies], 'Convergence Epoch average': [np.mean(x) for x in convergence_epochs]})

    for i, accuracy in enumerate(accuracies):
        for j, acc in enumerate(accuracy):
            step_size_df.loc[i, 'Fold ' + str(j + 1)] = acc
            step_size_df.loc[i, 'Convergence Epoch ' + str(j + 1)] = convergence_epochs[i][j]
    step_size_df.to_csv('step_size.csv', index=False)
    return accuracies, convergence_epochs

    
#**************************************************************
# Logistic regression algorithm with the following options

def logistic_regression(X, y, step_size=0.01, epochs=2500, stochastic=False, regularization=False, lambda_=0.05, decay= False, decay_rate= 0.8):

    weights = np.zeros(X.shape[1] + 1) # initialize weights, w = [0] * (d + 1)
    X = np.c_[np.ones(X.shape[0]), X] # add bias term to X
    y = y.values
    y = np.where(y == 'Cammeo', 1, -1)
    E_min = np.inf
    best_weights = weights
    losses = []
    epsilon = 1e-5
    prev_E = np.inf
    E = np.inf
    convergenced_epoch = 0
    converged = False

    for t in range(epochs):
        prev_E = E
        E = 0
        for i in range(X.shape[0]):
            E += np.log(1  + np.exp(-y[i] * np.dot(weights, X[i])))
        E = E / X.shape[0]

        if abs(E - prev_E) < epsilon and not converged:
            convergenced_epoch = t
            converged = True


        losses.append(E)
        if E < E_min:
            E_min = E
            best_weights = weights


        gradient=0
        if stochastic:
            i = np.random.randint(X.shape[0])
            gradient = - (y[i] * X[i])/ (1 + np.exp(y[i] * np.dot(weights, X[i])))
            # Should decay be true in stochastic case?
            decay= True
            if decay and t % 100 ==0: # GD does not have decay option
                step_size *= decay_rate

        else:
            for i in range(X.shape[0]):
                gradient -= (y[i] * X[i])/ (1 + np.exp(y[i] * np.dot(weights, X[i])))

        if regularization:
            gradient += 2 * lambda_ * weights
    

        gradient = - gradient / np.linalg.norm(gradient)
        weights += step_size * gradient
    
    return weights, best_weights, E_min, losses, convergenced_epoch



def test(X, y, weights):
    X = np.c_[np.ones(X.shape[0]), X] # add bias term to X
    y = y.values
    y = np.where(y == 'Cammeo', 1, -1)
    h = sigmoid(np.dot(X, weights))
    h = np.where(h > 0.5, 1, -1)
    accuracy = np.sum(h == y) / y.size
    return accuracy




#********************** Deliverable 3, 4 **********************
#plot and compare acccuracy of GD and SGD
def compare_GD_SGD(X, y, lambda_=0.05):
   
    X, y = preprocess(X, y)
    train_X, test_X, train_y, test_y = split_data(X, y)

    # run configs
    configs = {'GD - no regularization': {'step_size': 0.01, 'epochs': 2500, 'stochastic': False, 'regularization': False, 'lambda_': lambda_},
            'GD - regularization':{'step_size': 0.01, 'epochs': 2500, 'stochastic': False, 'regularization': True, 'lambda_': lambda_},
            'SGD - no regularization':{'step_size': 0.01, 'epochs': 2500, 'stochastic': True, 'regularization': False, 'lambda_': lambda_},
            'SGD - regularization':{'step_size': 0.01, 'epochs': 2500, 'stochastic': True, 'regularization': True, 'lambda_': lambda_} }
    


    comparison_data = []
    all_losses = []
    for name, config in configs.items():
        start_time = time.time()
        weights, best_weights, E_min, losses, _ = logistic_regression(train_X, train_y, step_size=config['step_size'], epochs=config['epochs'], stochastic=config['stochastic'], regularization=config['regularization'], lambda_=config['lambda_'])
        end_time = time.time()
        all_losses.append(losses)
        test_accuracy = test(test_X, test_y, weights)
        train_accuracy = test(train_X, train_y, weights)
        print("Test Accuracy: ", test_accuracy)
        print("Train Accuracy: ", train_accuracy)
        comparison_data.append({'Config': name, 'Test Accuracy': test_accuracy, 'Train Accuracy': train_accuracy, 'Weights': weights, 'Times': end_time - start_time})

    plt.figure()
    for i, losses in enumerate(all_losses):
        if i == 0:
            plt.plot(losses, label='No Regularization, No Stochastic')
        elif i == 1:
            plt.plot(losses, label='Regularization, No Stochastic')
        elif i == 2:
            plt.plot(losses, label='No Regularization, Stochastic')
        else:
            plt.plot(losses, label='Regularization, Stochastic')
  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig('SGD-GD.png')

    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('SGD-GD.csv', index=False)

#********************** Deliverable 5 **********************

#Investigate the effect of the step size in SGD on convergence.
#compare different step sizes with the default step size explored

def investigate_step_size(X, y, lambda_=0.05):
    X, y = preprocess(X, y)
    train_X, test_X, train_y, test_y = split_data(X, y)

    step_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
    all_losses = []
    for step_size in step_sizes:
        weights, best_weights, E_min, losses, _ = logistic_regression(train_X, train_y, step_size=step_size, epochs=2500, stochastic=True, regularization=True, lambda_=lambda_, decay=True, decay_rate=0.9)
        all_losses.append(losses)

    plt.figure()
    for i, losses in enumerate(all_losses):
        plt.plot(losses, label='Step Size: ' + str(step_sizes[i]))
  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch for Different Step Sizes')
    plt.legend()
    plt.savefig('step_size_convergence.png')

  

if __name__ == "__main__":
    X = pd.read_csv('rice_cammeo_and_osmancik.csv', index_col=0)
    y = pd.read_csv('rice_cammeo_and_osmancik_targets.csv', index_col=0)
    args = argparse.ArgumentParser()
    args.add_argument("--deliverable", type=int, default=1)
    args = args.parse_args()
    if args.deliverable == 1:
        X, y = preprocess(X, y)
    elif args.deliverable == 2:
        determine_best_lambda(X, y)
    elif args.deliverable == 3 or args.deliverable == 4:
        compare_GD_SGD(X, y)
    elif args.deliverable == 5:
        investigate_step_size(X, y)
    else:
        print("Invalid deliverable number")


    
        
    

