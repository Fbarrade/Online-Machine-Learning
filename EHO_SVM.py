import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define the target function class
class TargetFunction:
    def __init__(self, X_train, y_train, X_valid, y_valid, maxIter=50):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.maxIter = maxIter

    def function(self, position):
        # Train an SVM classifier with the given parameters
        C_value = max(0.001, position[0])  # Ensure 'C' is within the valid range
        clf = SVC(C=C_value, kernel='linear')
        clf.fit(self.X_train, self.y_train)

        # Predict on the validation set
        y_pred = clf.predict(self.X_valid)
        
        # Evaluate accuracy on the validation set
        accuracy = accuracy_score(self.y_valid, y_pred)
        
        return accuracy

# Define the ElephantAgent class
class ElephantAgent:
    def __init__(self, targetFunction):
        self.targetFunction = targetFunction
        self.position = np.random.uniform(0, 1, size=(1,))  # Initial position
        self.fitness = self.targetFunction.function(self.position)
        self.isAdult = False
        self.isMatriarch=False

    def update_position(self, newPosition):
        self.position = newPosition
        self.fitness = self.targetFunction.function(self.position)

# Define the ElephantClan class
class ElephantClan:
    def __init__(self, targetFunction, alpha, beta, N_elephants=100):
        self.alpha = alpha
        self.beta = beta
        self.targetFunction = targetFunction
        self.elephants = [ElephantAgent(targetFunction) for _ in range(N_elephants)]
        self.bestPosition = None
        self.bestFitness = None
        self.matriarch = self.get_matriarch()  # Initialize matriarch here
        self.adult = self.get_adult()  # Initialize adult here
        self.gamma = 0.8  # Add gamma attribute

    def get_matriarch(self):
        return max(self.elephants, key=lambda elephant: elephant.fitness)

    def get_adult(self):
        return min(self.elephants, key=lambda elephant: elephant.fitness)

    def get_center(self):
        return np.mean([elephant.position for elephant in self.elephants if not elephant.isAdult], axis=0)

    def update_positions(self):
        for elephant in self.elephants:
            if not elephant.isAdult:
                if elephant.isMatriarch:
                    new_position=new_position = self.beta * self.get_center()
                    elephant.update_position(new_position)
                else:
                    new_position = elephant.position + self.alpha * (self.matriarch.position - elephant.position) 
                    elephant.update_position(new_position)

    def run(self, verbose=False):
        for _ in tqdm(range(self.targetFunction.maxIter), disable=not verbose):
            self.update_positions()
            self.matriarch.isMatriarch = False
            self.matriarch = self.get_matriarch()
            self.adult.isAdult = True
            self.adult.position = np.random.uniform(0, 1)

            # Update the best position and fitness found by this clan
            self.bestPosition = self.matriarch.position
            self.bestFitness = self.matriarch.fitness


# Define the EHOptimizer class
class EHOptimizer:
    def __init__(self, X_train, y_train, X_valid, y_valid, N_clans=5, N_elephants=10):
        self.N_clans = N_clans
        self.clans = [ElephantClan(TargetFunction(X_train, y_train, X_valid, y_valid), 
                                    alpha=np.random.uniform(0, 1), beta=np.random.uniform(0, 1)) 
                      for _ in range(N_clans)]
        self.bestPosition = None
        self.bestFitness = None

    def find_optimum(self, verbose=False):
        for clan in self.clans:
            clan.run(verbose=verbose)
        
        best_clan = max(self.clans, key=lambda c: c.bestFitness)
        self.bestPosition = best_clan.bestPosition
        self.bestFitness = best_clan.bestFitness


if __name__ == "__main__":
    data = pd.read_csv('BankNote_Authentication.csv').sample(frac=1).reset_index(drop=True)
    X = data.drop('class', axis=1).values
    y = data['class'].values

    # Normalize features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Split the dataset into training and testing sets
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]

    # Instantiate the optimizer
    optimizer = EHOptimizer(X_train, y_train, X_test, y_test)
    
    # Find the optimum
    optimizer.find_optimum(verbose=True)
    
    # Get the best position and fitness
    best_position = optimizer.bestPosition
    best_fitness = optimizer.bestFitness
    
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
