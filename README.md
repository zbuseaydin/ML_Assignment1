# CMPE462 - Machine Learning Assignment 1
## Part 1: Perceptron Learning Algorithm (PLA)
### Dependencies
- numpy
- matplotlib.pyplot
### Data
Data is under the "pla_data" directory which is in the same directory as the script.
### Running the Code
The script can be run with the following command: python3 pla.py
### Output
The script runs the PLA algorithm on both small and large datasets with 10 different random weight initializations and reports various information on the file stats.txt under the pla_results folder. The file stats.txt contains detailed information on the calculations for the small dataset like initial and final weights, number of iterations for convergence, the accuracy on test data and the equation of the decision boundary. It also reports the average number of iterations for convergence and the average accuracy for both datasets.

## Part 2: Logistic Regression
### Dependencies
- numpy
- pandas
- matplotlib.pyplot
- time
- argparse
### Data
Data is under the "logistic_regression_data" directory which is in the same directory as the script.
### Running the Code
The script implements logistic regression with variations. The code produces output for the following cases:
- Finding Best Lambda (Deliverable 2):```python3 logistic_regression.py -d 2```
- Comparing CD and SGD( Deliverable 3-4): ```python3 logistic_regression.py -d 3``` or ```python3 logistic_regression.py -d 4```
- Investigate Step Size (Deliverable 5): ```python3 logistic_regression.py -d 5```
- Train Logistic Regression with Breast Cancer Data: ```python3 logistic_regression.py -d 0```
### Output
- Finding Best Lambda (Deliverable 2): Creates lambda.xlsx that includes the accuracies for different lambda values by using 5-fold cross validation.
- Comparing CD and SGD (Deliverable 3-4): Creates SGD-GD.xlsx and SGD-GD.png files. Excel file includes accuracies of configurations of logsitic regression. Image is the plot of the loss vs epoch in training.
- Investigating step size (Deliverable 5): Creates step_size.xlsx that includes the information of searchin the best step size by using 5-fold cross validation again. Table also includes the convergence epoch for each step size. step_size_convergence.png file that plots loss of SGD with different initial step size.
- Train Logistic Regression with Breast Cancer Data: It prints out the train and test acuracies.


## Part 3: Naive Bayes
### Dependencies
- math
### Data
Data is under the "breast+cancer+wisconsin+diagnostic" directory which is in the same directory as the script.
### Running the Code
The script can be run with the following command:
```python3 naive_bayes.py```
### Output
The script prints out the train accuracy and test accuracy of the naive Bayes classifier.
