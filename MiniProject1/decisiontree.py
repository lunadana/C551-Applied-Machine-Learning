import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
The following cost functions were taken from the class website code (colab):
    1 - cost_misclassification(labels)
    2 - cost_entropy(labels)
    3 - def cost_gini_index(labels)
"""


# computes misclassification cost by subtracting the maximum probability of any class
def cost_misclassification(labels):
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    return 1 - np.max(class_probs)


# computes entropy of the labels by computing the class probabilities
def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]  # this steps is remove 0 probabilities for removing numerical issues while computing log
    return -np.sum(class_probs * np.log(class_probs))   # expression for entropy -\sigma p(x)log[p(x)]


# computes the gini index cost
def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))   # expression for gini index 1-\sigma p(x)^2


def merge_features_labels(features, labels):
    """
    Helper function that re-merges X_train & Y_train into one DataFrame data_train
    :param features: Numpy.ndarray - X_train
    :param labels: Numpy.ndarray - Y_train
    :return: Pandas.DataFrame - data_train
    """
    data = pd.DataFrame(features)
    labels = pd.DataFrame(labels)
    data['Labels'] = labels
    return data


def train_test_split(X, Y, test_size=0.2, rand_seed=None):
    num_instances = X.shape[0]
    num_instances_train = int((1-test_size)*num_instances)

    if rand_seed is None:
        np.random.seed(random.randint(0, 123456789))
    else:
        np.random.seed(rand_seed)

    inds = np.random.permutation(num_instances)
    # train-test split
    x_train, y_train = X[inds[:num_instances_train]], Y[inds[:num_instances_train]]
    x_test, y_test = X[inds[num_instances_train:]], Y[inds[num_instances_train:]]
    return x_train, x_test, y_train, y_test


def evaluate_acc(y_pred, y_test):
    """
    Evaluates the model's accuracy as (correct / total) * 100%
    :param y_pred: Numpy.ndarray - the [1 x n] array of predictions given from the model
    :param y_test: Numpy.ndarray - the [n x 1] array of true labels from the dataset
    :return: float - the accuracy of the model
    """
    tot = 0
    correct = 0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == int(y_test[i][0]):
            correct += 1
        tot += 1
    return correct / tot * 100


def get_best_acc_cols(x, y, num_iterations=10):
    """
    Helper function to determine the average accuracy over num_iterations iterations for every possible pair of features
    in a dataset x on a given model
    :param x: Pandas.DataFrame - the processed/cleaned 2D-array of features
    :param y: Pandas.DataFrame - the processed/cleaned 2D-array of labels
    :param model: the machine learning model that implements the fit & predict function as specified
    :param num_iterations: int - the # of iterations w/ random data split for each feature pair
    :return: tuple - (((INDEX BEST FEATURE 1, INDEX BEST FEATURE 2), ACCURACY), COLS)
                        where COLS is the dictionary storing the average accuracies for all feature pairs
    """
    # stores the feature pair with the best accuracy as ((INDEX BEST FEATURE 1, INDEX BEST FEATURE 2), ACCURACY)
    best_acc = ((0, 1), 0)

    # stores the average accuracies for all feature pairs as
    # { (INDEX FEATURE 1, INDEX FEATURE 2): ACCURACY1, (INDEX FEATURE 1, INDEX FEATURE 3): ACCURACY2, ... }
    cols = {}
    num_features = x.shape[1]
    r = np.random.permutation(num_iterations)

    # iterates over the number of features - 1 to avoid the case where a = feature x & b = feature x
    for a in range(num_features - 1):
        # iterates over the number of features after a to ensure a != b
        for b in range(a + 1, num_features):
            accuracy_sum = 0    # stores the sum of accuracies for the current feature pair (a,b)

            # calculates the accuracy over num_iterations trials & updates the average for this feature pair (a,b)
            for rand in r:
                X = x[[x.columns[a], x.columns[b]]].values    # filters the features to the 2 columns at index a & b
                Y = y.values.reshape(-1, 1)

                # splits the data into training and testing sets
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, rand_seed=((rand+1)**3))

                model = DecisionTree(max_depth=20, min_instances=2, cost_function=cost_entropy)
                model.fit(x_train, y_train)    # fits the model on the data of the current feature pair
                y_pred = model.predict(x_test)    # gets the predictions made from the model on the test set
                accuracy_sum += evaluate_acc(y_pred, y_test)    # updates the accuracy sum for the current feature pair

            avg_acc = accuracy_sum / num_iterations    # computes the average of the accuracies


            # if the current feature pair (a,b) yields a better average accuracy than the current best --> update
            if avg_acc > best_acc[1]:
                best_acc = ((a, b), avg_acc)
            cols[(a, b)] = avg_acc    # stores the current feature pair's average accuracy in the dictionary
            print(cols)
    return best_acc, cols


def best_split(data, cost_function):
    """
    Determines the best split of the data by iterating through all possible split values & calculates the cost of that
    split using the given parameter cost_function. The best split is the split with the lowest cost.
    :param data: Pandas.DataFrame - the data to be split
    :param cost_function: function - the cost function of the decision tree that computes a split cost
    :return: dict - {'cost': best split cost, 'feature': best split feature, 'test': best split value,
                     'data': {'left': left split data, 'right': right split data}}
    """

    # initializes the dictionary storing the information of the best data split to be returned
    split = {'cost': np.inf, 'feature': None, 'test': None, 'data': {'left': None, 'right': None}}

    # iterates through each possible feature column in the data (-1 because of assumed last column being the labels)
    for feature_index in range(data.shape[1] - 1):
        feature_vector = data.values[:, feature_index]  # stores the column of the feature at feature_index
        feature_tests = np.unique(feature_vector)   # stores all unique values for that feature

        # iterates through each possible value for a test on the current feature
        for test in feature_tests:
            # split the data into left and right children
            data_left = data.loc[data[data.columns[feature_index]] <= test]  # stores the data that is <= the curr test
            data_right = data.loc[data[data.columns[feature_index]] > test]  # stores the data that is > the curr test

            if data_left.shape[0] >= 0 and data_right.shape[0] >= 0:
                cost_left = cost_function(data_left.iloc[:, -1])    # computes cost of the left split
                cost_right = cost_function(data_right.iloc[:, -1])  # computes cost of the right split

                # computes the total cost of the split by weighing the left and right costs accordingly
                cost_weighted = ((data_left.shape[0] * cost_left) + (data_right.shape[0] * cost_right)) / data.shape[0]

                # if the current total cost is lower than our best cost found so far --> update the best cost to the
                # current split
                if cost_weighted < split['cost']:
                    split = {'cost': cost_weighted, 'feature': feature_index, 'test': test,
                             'data': {'left': data_left, 'right': data_right}}
    return split


class Node:
    """
    Stores information contained at a node from the decision tree
    """
    def __init__(self, feature=None, test=None, left_child=None, right_child=None, label=None):
        """
        Initializes a Node object & sets the given parameters
        :param feature: int - the index of the feature from the dataset that this node will test on
        :param test: the split value on the feature stored in self.feature
        :param left_child: Node - the left child of this node. Can be None if self.is_leaf()
        :param right_child: Node - the right child of this node. Can be None if self.is_leaf()
        :param label: int - if self.is_leaf(), it stores the leaf's label. None otherwise.
        """
        self.feature = feature
        self.test = test
        self.left_child = left_child
        self.right_child = right_child
        self.label = label

    def is_leaf(self):
        """
        Determines if self is a leaf node in the decision tree by checking if self has no children
        :return: True if self is a leaf node & False otherwise
        """
        return self.left_child is None and self.right_child is None

    def get_label(self):
        """
        Checks if self is a leaf node & gives the label if so
        :return: label stored in this leaf node, None otherwise
        """
        if self.is_leaf():
            return self.label


class DecisionTree:
    """
    Decision Tree algorithm is implemented as follows:

        Calling DecisionTree.fit() will build the decision tree using the private DecisionTree._build():
            1 - if there are enough instances in the data and not passed the max depth of the tree,
                it splits the data according to the greedy split function best_split() which iterates through
                all possible split values & calculates the cost of that split using the given parameter cost_function.
                The function returns the split with the minimal cost.
            2 - it, recursively, builds the left subtree and right subtree from the current node following step 1 on
                every new node until it reaches a leaf node. At leaf nodes, only the label is stored in the node.
                This is the classification class when using the decision tree.

        Calling DecisionTree.predict() will, for each instance in the test data, go through each node of the decision
        tree following the split test values to go to the left/right child for each node until it reaches a leaf node
        at which that instance's label is determined.
    """
    def __init__(self, max_depth=3, min_instances=1, cost_function=cost_gini_index):
        """
        Initializes the decision tree's root as None and stores the given (optional) parameters
        :param max_depth: int - the maximum depth of the tree
        :param min_instances: int - the minimum number of instances required to split a node
        :param cost_function: function - a cost function that takes in labels & computes the split cost
        """
        self.root = None
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.cost_function = cost_function

    def _build(self, data, curr_depth=0):
        """
        Recursively builds the entire decision tree given the data as follows:
            1 - if there are enough instances in the data and not passed the max depth of the tree,
                it splits the data according to the greedy split function best_split()
            2 - it, recursively, builds the left subtree and right subtree from the current node following step 1 on
                every new node until it reaches a leaf node. At leaf nodes, only the label is stored in the node.
                Leaf nodes are the classification class when using the decision tree.
        :param data: Pandas.DataFrame - the training data to build our decision tree on
        :param curr_depth: int - the current depth of the tree
        :return: Node
        """
        features = data.values[:, :-1]          # stores the 2D array of features (assumes last col are labels)
        num_instances = features.shape[0]       # stores number of instances of the data (num of rows)

        # if there are enough instances in the data and not passed the max depth of the tree --> split data
        if num_instances > self.min_instances and curr_depth < self.max_depth:
            split = best_split(data, self.cost_function)             # gets the split with minimum cost (best split)

            if np.isinf(split['cost']) or split['feature'] is None:  # this shouldn't happen w/ reasonable min_instances
                return

            # recursively builds the left and right subtrees of the current node
            left_child = self._build(split['data']['left'], curr_depth+1)
            right_child = self._build(split['data']['right'], curr_depth+1)

            return Node(feature=split['feature'], test=split['test'], left_child=left_child, right_child=right_child)

        # reached a leaf node so the label of this node is determined by the most frequent label in the data (mode)
        if num_instances > 0:
            leaf_label = int(data['Labels'].mode().values[0])
            return Node(label=leaf_label)

    def fit(self, train_features, train_labels):
        """
        Builds the decision tree with the given training data
        :param train_features: Numpy.ndarray - the training features (X_train)
        :param train_labels: Numpy.ndarray - the training labels (Y_train)
        :return: DecisionTree - the tree that was just built, self
        """
        self.root = self._build(merge_features_labels(train_features, train_labels))

        nodes = [self.root]
        while len(nodes) > 0:
            curr_node = nodes.pop()
            if curr_node is not None:
                nodes.append(curr_node.right_child)
                nodes.append(curr_node.left_child)
                if not curr_node.is_leaf():
                    if curr_node.left_child is None:
                        curr_node.left_child = curr_node.right_child
                    elif curr_node.right_child is None:
                        curr_node.right_child = curr_node.left_child

        return self

    def predict(self, test_data):
        """
        For each instance in the test data, it goes through each node of the decision tree following the split test
        values to go to the left/right child for each node until it reaches a leaf node at which that instance's label
        is determined.
        :param test_data: Pandas.DataFrame - the test data for which it will classify each instance
        :return: Numpy.ndarray - the array of predictions for each instance in the test data
        """
        predictions = np.zeros(test_data.shape[0])      # initializes the array of predictions as zeros

        # iterates through each instance and determines that instance's label by following the splits of the dec tree
        for index, instance in enumerate(test_data):
            curr_node = self.root
            while not curr_node.is_leaf():
                if instance[curr_node.feature] <= curr_node.test:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child

            # at end of the while loop for each instance, curr_node stores the leaf node that has the predicted label
            # for the instance, so stores that in the array of predictions for that instance
            predictions[index] = curr_node.get_label()
        return predictions


def get_n_most_correlated_features_w_label(data, labels, n):
    """
    Gets the n most (+/-) correlated features with with labels
    :param data: Pandas.DataFrame - the 2D features array
    :param labels: Pandas.DataFrame - the array of labels
    :param n: int - the # of the most correlated features (n <= # of features)
    :return: list - the indices of the n most correlated features to the labels in descending order
    """
    n_highest_corr_features = []
    features = {}
    for i in range(data.shape[1]):
        features[i] = abs(labels.corr(data.iloc[:, i]))

    # finds the n features with the highest correlation
    for i in range(n):
        # stores the current feature with the highest corr as a tuple (feature index, correlation)
        max_corr_feature = (-1, -1)
        for feature_index in features:
            if features[feature_index] > max_corr_feature[1]:
                max_corr_feature = (feature_index, features[feature_index])

        n_highest_corr_features.append(max_corr_feature[0])
        del features[max_corr_feature[0]]

    return n_highest_corr_features


# main function used for various testing of models: accuracy, decision boundaries, graphing, etc.
if __name__ == '__main__':
    # --- DATA PREPROCESSING ---

    data = pd.read_csv(r"data/hepatitis_clean.csv", header=None)
    data.drop(index=data.index[0], axis=0, inplace=True)
    for col in data.columns:
        data[col] = data[col].astype(float)
    x, y = data.iloc[:, 2:], data.iloc[:, 1]

    colors = {1: 'red', 2: 'green'}
    selected_cols_index = get_n_most_correlated_features_w_label(x, y, 10)
    #selected_cols_index = [2, 14, 15, 13, 12]

    """
    # --- AVG MODEL PERFORMANCE ---
    
    selected_cols = [x.columns[i] for i in selected_cols_index]
    
    X = x[selected_cols].values
    Y = y.values.reshape(-1, 1)
    
    xvals = []
    yvals = []
    
    num_of_runs = 100
    
    cost_fns = [cost_entropy]
    max_depths = [5]
    min_instances = [50]
    
    for min_instance in min_instances:
        for depth in max_depths:
            for cost_fn in cost_fns:
                avg = 0
                for i in range(num_of_runs):
                    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, rand_seed=None)
                    model = DecisionTree(max_depth=depth, min_instances=min_instance, cost_function=cost_fn)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
    
                    #xvals.append(i)
                    #yvals.append(evaluate_acc(y_pred, y_test))
    
                    avg += evaluate_acc(y_pred, y_test)
                print(f"Depth={depth}, CostFn={cost_fn.__name__} | Accuracy={avg/num_of_runs}%")
    
    # plt.scatter(xvals, yvals)
    # plt.xlabel('Min Instances')
    # plt.ylabel('Model accuracy (%)')
    # plt.title('Model Accuracy vs. Min Instances')
    # plt.show()
    
    print("Average model accuracy on test set: " + str(avg/num_of_runs) + "%")
    """

    #"""
    # --- VIEWING MODEL PERFORMANCE (graph & decision boundary based on feature 1&2 index - f1, f2) ---

    selected_cols = [x.columns[i] for i in selected_cols_index]

    #best_acc = get_best_acc_cols(x, y, 10)[0]
    #f1 = best_acc[0][0]
    f1_index_original = 16
    f1_index_new = list(selected_cols_index).index(f1_index_original)
    f1_name = x.columns[f1_index_original]
    f1_axistitle = "Albumin"

    #f2 = best_acc[0][1]
    f2_index_original = 17
    f2_index_new = list(selected_cols_index).index(f2_index_original)
    f2_name = x.columns[f2_index_original]
    f2_axistitle = "Protime"

    X = x[selected_cols].values
    Y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, rand_seed=None)
    model = DecisionTree(max_depth=20, min_instances=5, cost_function=cost_misclassification)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    x_test_correct = []
    x_test_incorrect = []
    y_pred_correct = []
    y_pred_incorrect = []

    for i in range(len(y_pred)):
        if int(y_pred[i]) == int(y_test[i][0]):
            x_test_correct.append(x_test[i])
            y_pred_correct.append(y_test[i][0])
        else:
            y_pred_incorrect.append(y_test[i][0])
            x_test_incorrect.append(x_test[i])

    x_test_correct = np.array(x_test_correct)
    x_test_incorrect = np.array(x_test_incorrect)
    y_pred_correct = np.array(y_pred_correct)
    y_pred_incorrect = np.array(y_pred_incorrect)

    c_train = [colors[int(i)] for i in y_train]
    c_correct = [colors[int(i)] for i in y_pred_correct]
    c_incorrect = [colors[int(i)] for i in y_pred_incorrect]

    try:
        plt.scatter(x_train[:, f1_index_new].astype('float'), x_train[:, f2_index_new].astype('float'), c=c_train, marker='o', alpha=0.4,
                    label='Train')
        plt.scatter(x_test_correct[:, f1_index_new].astype('float'), x_test_correct[:, f2_index_new].astype('float'), c=c_correct,
                    label='Correct')
        plt.scatter(x_test_incorrect[:, f1_index_new].astype('float'), x_test_incorrect[:, f2_index_new].astype('float'), c=c_incorrect,
                    marker='x', label='Misclassified')
    except:
        plt.scatter([], [], marker='x', label='Misclassified')

    plt.legend()
    plt.show()

    print("The model accuracy on test set: " + str(evaluate_acc(y_pred, y_test)) + "%")

    #  --- DECISION BOUNDARY SECTION ---

    X = x[[f1_name, f2_name]].values
    Y = y.values.reshape(-1, 1)

    x_train_featurepair = x_train[:, [f1_index_new, f2_index_new]]
    x_test_featurepair = x_test[:, [f1_index_new, f2_index_new]]

    model = DecisionTree(max_depth=20, min_instances=5, cost_function=cost_misclassification)
    model.fit(x_train_featurepair, y_train)

    c_train = [colors[int(i)] for i in y_train]

    granularity = 200
    x0v = np.linspace(float(x.iloc[:, f1_index_original].min()), float(x.iloc[:, f1_index_original].max()), granularity)
    x1v = np.linspace(float(x.iloc[:, f2_index_original].min()), float(x.iloc[:, f2_index_original].max()), granularity)
    x0, x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(), x1.ravel())).T

    y_pred = model.predict(x_test_featurepair)
    y_pred_all = model.predict(x_all)

    try:
        plt.scatter(x_all[:, 0], x_all[:, 1], c=[colors[int(i)] for i in y_pred_all], marker='.', alpha=0.05)
        plt.scatter(x_train[:, f1_index_new].astype('float'), x_train[:, f2_index_new].astype('float'), c=c_train)
    except:
        pass

    plt.xlabel(f1_axistitle)
    plt.ylabel(f2_axistitle)
    plt.title("Decision Boundary for Detecting DIE/LIVE in the Hepatitis Dataset")

    plt.show()
    print(f"The model accuracy on test set using features {f1_axistitle} & {f2_axistitle}: " + str(evaluate_acc(y_pred, y_test)) + "%")
    #"""
