import numpy as np
from collections import defaultdict
from collections import Counter
import pandas as pd

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X,y):
        self.root = self._grow_tree(X,y)
        
    def _grow_tree(self,X,y, depth = 0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)
        
        #Pre pruning
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value = Counter(y).most_common(1)[0][0])
        
        best_feature, best_threshold = self._best_split(X, y, num_features)
        
        if best_feature is None:
            return Node(value = Counter(y).most_common(1)[0][0])
        
        left_index = X[:, best_feature] < best_threshold
        right_index = ~left_index 
        left_child = self._grow_tree(X[left_index],y[left_index],depth+1)
        right_child = self._grow_tree(X[right_index],y[right_index], depth+1)        
        
        return Node(feature = best_feature, threshold = best_threshold, left = left_child, right = right_child)
    
    def _best_split(self,X, y, num_features):
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            threshold = np.unique(X[:,feature])
            for threshold in threshold:
                gain = self._information_gain(y,X[:,feature],threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    
    def _information_gain(self, y, X_column, threashold):   
        parent_entropy = self._entropy(y)
        left_y, right_y = y[X_column < threashold], y[X_column >= threashold]
        n,n_left, n_right = len(y), len(left_y), len(right_y)
        if n_left == 0 or n_right == 0:
            return 0
        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy
        
    
    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / np.sum(counts)
        return -np.sum([p*np.log2(p) for p in probabilities if p > 0])
    
    def post_prune(self, node, X_val, y_val):
        self.prune_node(self.root, X_val, y_val)
        
    def prune_node(self, node, X_val, y_val):
        if node is None or node.is_leaf():
            return
        
        # Recursively prune left and right subtrees
        self.prune_node(node.left, X_val, y_val)
        self.prune_node(node.right, X_val, y_val)

        # If both children are leaves, consider pruning
        if node.left.is_leaf() and node.right.is_leaf():
            left_leaf_value = node.left.value
            right_leaf_value = node.right.value

            # Majority class if we prune
            majority_class = Counter(y_val).most_common(1)[0][0]

            # Evaluate accuracy before pruning
            y_pred_before = self.prediction(X_val)
            accuracy_before = np.mean(y_pred_before == y_val)

            # Temporarily prune the node
            node.value = majority_class
            node.left = None
            node.right = None

            # Evaluate accuracy after pruning
            y_pred_after = self.prediction(X_val)
            accuracy_after = np.mean(y_pred_after == y_val)

            # Keep the pruning only if it does not reduce accuracy
            if accuracy_after < accuracy_before:
                node.value = None
                node.left = Node(value=left_leaf_value)
                node.right = Node(value=right_leaf_value)
        
    
    def prediction(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    
if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn import tree #this will be the sklearn decision tree
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    
    '''
    Will begin with the sklearn decsision tree first
    '''
    wine = load_wine()#loads all the data into wine
    X, y = wine.data, wine.target#seperate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    print("The following is the sklearn library implementation of the Decision tree")
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)#fits the model iwth the data
    predict = clf.predict(X_test)
    print("The following is the version with no pruning")
    print(predict)
    print(f"Accuracy: {accuracy_score(y_test, predict):.2f}")
    print(f"Precision: {precision_score(y_test, predict, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, predict, average='weighted'):.2f}")
    print(f"F1-score: {f1_score(y_test, predict, average='weighted'):.2f}\n")
    
    pruned_tree = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=10, min_samples_leaf=5)#Does Pre-Pruning by limiting the amount of depth of the tree by limiting the tree before training
    pruned_tree = pruned_tree.fit(X_train, y_train)
    path = pruned_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Train a series of decision trees with different alpha values
    pruned_models = []
    for ccp_alpha in ccp_alphas:
        pruned_model = tree.DecisionTreeClassifier(criterion="entropy", ccp_alpha=ccp_alpha)
        pruned_model.fit(X_train, y_train)
        pruned_models.append(pruned_model)

    # Find the model with the best accuracy on test data
    best_accuracy = 0
    best_pruned_model = None
    for pruned_model in pruned_models:
        accuracy = pruned_model.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pruned_model = pruned_model
    
    
    predict2 = best_pruned_model.predict(X_test)
    print("The following is the version with pre and post pruning")
    print(predict2)
    
    
    # Model Accuracy after pruning
    accuracy_after_pruning = best_pruned_model.score(X_test, y_test)
    print(f"Accuracy: {accuracy_score(y_test, predict2):.2f}")
    print(f"Precision: {precision_score(y_test, predict2, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, predict2, average='weighted'):.2f}")
    print(f"F1-score: {f1_score(y_test, predict2, average='weighted'):.2f}\n")
    
    
    print(f"Depth of unpruned tree: {clf.get_depth()}")
    print(f"Depth of best pruned tree: {best_pruned_model.get_depth()}\n")
       
    
    '''
    This is the OOP version of the Decision Tree implementation
    '''
    
    
    tree = DecisionTree(max_depth = 3) 
    tree.fit(X_train,y_train)
    prediction = tree.prediction(X_test)
    
    print("The following is the OOP implementation of the Decision tree with only pre pruning \n")
    
    print (prediction)
    print("\nCustom Decision Tree Metrics with no pruning:")
    print(f"Accuracy: {accuracy_score(y_test, prediction):.2f}")
    print(f"Precision: {precision_score(y_test, prediction, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, prediction, average='weighted'):.2f}")
    print(f"F1-score: {f1_score(y_test, prediction, average='weighted'):.2f} \n")
    
    print("The following is the OOP version of the Decision Tree with pre and post pruning")
    
    #seperates the data into 60% training data, 20% validation data for pruning and 20% for testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    
    tree.post_prune(tree.root,X_val,y_val)
    prediction2 = tree.prediction(X_test)
    
    
    print (prediction2)
    print("\nCustom Decision Tree Metrics with pruning:")
    print(f"Accuracy: {accuracy_score(y_test, prediction2):.2f}")
    print(f"Precision: {precision_score(y_test, prediction2, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, prediction2, average='weighted'):.2f}")
    print(f"F1-score: {f1_score(y_test, prediction2, average='weighted'):.2f} \n")
    
    
    
    
          