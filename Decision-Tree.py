import numpy as np
from collections import defaultdict
from collections import Counter

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
        
        return Node(feature = best_feature, threashod = best_threshold, left = left_child, right = right_child)
    
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
    
    def _variance(self, y):
        return np.var(y)
        
    
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
    #from matplotlib.pyplot import plt
    
    '''
    Will begin with the sklearn decsision tree first
    '''
    wine = load_wine()

    X, y = wine.data, wine.target
    
    
    clf = tree.DecisionTreeClassifier()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    clf = clf.fit(X_train, y_train)#fits the model iwth the data
    predict = clf.predict(X_test)
    print(predict)
    accuracy = np.mean(predict == y_test)
    print(F"Accuracy: {accuracy:.2f}")
    
    
    #tree.plot_tree(clf)
    
    '''
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    tree = DecisionTree(max_depth = 3) 
    tree.fit(X_train,y_train)
    prediction = tree.prediction(X_test)
    
    print (prediction)
    accuracy = np.mean(prediction == y_test)
    print(F"Accuracy: {accuracy:.2f}")
    '''
    
    
    
    
    
    
    
        