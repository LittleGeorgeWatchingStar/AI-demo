from random_tree import RandomTree
import numpy as np
from scipy import stats

class RandomForest:

    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.trees = []
        for m in range(self.num_trees):
            print("Fitting tree %02d/%d..." % (m+1,self.num_trees))
            tree = RandomTree(max_depth = self.max_depth)
            tree.fit(X,y)
            self.trees.append(tree)
            
    def predict(self, X):
        t = X.shape[0]
        yhats = np.ones((t,self.num_trees), dtype=np.uint8)

        # Predict using each model
        for m in range(self.num_trees):
            yhats[:,m] = self.trees[m].predict(X)

        # Take the most common label
        return stats.mode(yhats, axis=1)[0].flatten()