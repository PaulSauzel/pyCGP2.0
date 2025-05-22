from abc import ABC, abstractmethod
import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold

class Evaluator(ABC): #Abstract class for evaluator
    @abstractmethod
    def evaluate(self, genome):
        pass





class EvaluatorSin(Evaluator): #Evaluator for the sin function
    def __init__(self, input_range=(-1, 1), num_points=100):
        self.inputs = np.linspace(input_range[0], input_range[1], num_points) #Generate 100 points between -1 and 1
        self.targets = np.sin(self.inputs)


    def evaluate(self, genome,generation):
        predictions = [genome.get_value([x])[0] for x in self.inputs] 

        from sklearn.metrics import r2_score

        r2 = r2_score(self.targets, predictions)
        return r2 

    # try to add a graph to represent fitness over time (convergency grap)





#IN DEVELOPMENT
class Binary_Classifier(Evaluator):

    def __init__(self, X, y, test_size=0.2, random_state=42, threshold=0.5, cv = False):
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.threshold = threshold
        self.last_train_accuracy = 0.0
        self.last_test_accuracy = 0.0
        self.cv = cv  # Flag to indicate if cross-validation is used
 
    def evaluate(self, genome,generation):
        if self.cv:
            return self.evaluate_cv(genome, k=5)
        else:
            return self.evaluate_no_cv(genome)
        
    def evaluate_no_cv(self,genome):
        train_preds = []
        for x in self.X_train:
            output_value = genome.get_value(x)[0]
            predicted = 1 if output_value > self.threshold else 0
            train_preds.append(predicted)
        train_preds = np.array(train_preds).flatten()
        y_train_flat = np.array(self.y_train).flatten()
        self.last_train_accuracy = accuracy_score(y_train_flat, train_preds)

        # Predict on test set
        test_preds = []
        for x in self.X_test:
            output_value = genome.get_value(x)[0]
            predicted = 1 if output_value > self.threshold else 0
            test_preds.append(predicted)
        test_preds = np.array(test_preds).flatten()
        y_test_flat = np.array(self.y_test).flatten()
        self.last_test_accuracy = accuracy_score(y_test_flat, test_preds)


        return self.last_test_accuracy  # Use test accuracy as fitness '''
    

    def evaluate_cv(self, genome, k):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies = []
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.X_train)):
            X_fold_train = self.X_train[train_index]
            y_fold_train = self.y_train[train_index]
            X_fold_test = self.X_train[test_index]
            y_fold_test = self.y_train[test_index]

            # Predict on the test fold
            fold_preds = []
            for x in X_fold_test:
                output_value = genome.get_value(x)[0]
                predicted = 1 if output_value > self.threshold else 0
                fold_preds.append(predicted)

            fold_preds = np.array(fold_preds).flatten()
            y_fold_test = np.array(y_fold_test).flatten()
            acc = accuracy_score(y_fold_test, fold_preds)
            accuracies.append(acc)

        # Average accuracy over all folds
        mean_acc = np.mean(accuracies)
        self.last_train_accuracy = mean_acc  # We use training data split for cross-val, so this becomes our new metric
        self.last_test_accuracy = self.last_train_accuracy  # Optional: could still keep a separate real test set

        return self.last_train_accuracy   