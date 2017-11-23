import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import utils

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from knn import KNN, CNN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["1.1", "2", "2.2", "2.3", "2.4", "3", "3.1", "3.2", "4.1", "4.2", "5"])

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Q1.1 - This should print the answers to Q 1.1

        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")

        # part 1: min, max, mean, median and mode
        results = ("Min: %.3f, Max: %.3f, Mean: %.3f, Median: %.3f, Mode: %.3f" %
                   (np.min(X), np.max(X), np.mean(X), np.median(X), utils.mode(X)))
        print(results)

        # part 2: quantiles
        print("quantiles: %s" % np.percentile(X, [10, 25, 50, 75, 90]))

        # part 3: maxMean, minMean, maxVar, minVar
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)

        results = ("maxMean: %s, minMean: %s, maxVar: %s, minVar: %s" %
                   (names[np.argmax(means)], names[np.argmin(means)],
                    names[np.argmax(variances)], names[np.argmin(variances)]))

        # part 4: correlation between columns
        corr = np.corrcoef(X.T)

        # least correlated
        c_min = np.min(corr)
        c_least = np.where(corr == c_min)

        # most correlated
        c_max = np.max(corr - np.identity(X.shape[1]))
        c_most = np.where(corr == c_max)

        print("most correlated is between %s and %s\n"
              "least correlated is between %s and %s" %
              (names[c_most[0]], names[c_most[1]],
               names[c_least[0]], names[c_least[1]]))

    elif question == "2":

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with inequality rule error: %.3f"
              % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "2.2":
        # Q2.2 - Decision Stump

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q2_1_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "2.3":
        # Q2.3 - Decision Tree with depth 2

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
    
    elif question == "2.4":
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="mine")
        
        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        
        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q2_4_tree_errors.pdf")
        plt.savefig(fname)
        
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)


    elif question == "3":
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "3.1":
        # Q3.1 - Training and Testing Error Curves
        # 1. Load dataset
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        e_depth = 15
        s_depth = 1

        train_errors = np.zeros(e_depth - s_depth)
        test_errors = np.zeros(e_depth - s_depth)

        for i, d in enumerate(range(s_depth, e_depth)):
            print("\nDepth: %d" % d)

            model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)

            train_errors[i] = tr_error
            test_errors[i] = te_error

        x_vals = np.arange(s_depth, e_depth)
        plt.title("The effect of tree depth on testing/training error")
        plt.plot(x_vals, train_errors, label="training error")
        plt.plot(x_vals, test_errors, label="testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = os.path.join("..", "figs", "q3_1_trainTest.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        # This should show that the training error decreases to zero
        # The test error plateus at 0.1

    elif question == '3.2':
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        Xtrain = X[:n//2]
        ytrain = y[:n//2]
        Xvalid = X[n//2:]
        yvalid = y[n//2:]

        SWAP = 0
        if SWAP:
            Xtrain, Xvalid = Xvalid, Xtrain
            ytrain, yvalid = yvalid, ytrain

        depths = np.arange(1,15)

        train_errors = np.zeros(len(depths))
        val_errors = np.zeros(len(depths))

        for i, d in enumerate(depths):
            print("\nDepth: %d" % d)

            model = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=1)
            model.fit(Xtrain, ytrain)

            y_pred = model.predict(Xtrain)
            tr_error = np.mean(y_pred != ytrain)

            y_pred = model.predict(Xvalid)
            val_error = np.mean(y_pred != yvalid)
            print("Training error: %.3f" % tr_error)
            print("Validation error: %.3f" % val_error)

            train_errors[i] = tr_error
            val_errors[i] = val_error

        plt.title("The effect of tree depth on testing/training error")
        plt.plot(depths, train_errors, label="training error")
        plt.plot(depths, val_errors, label="testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = os.path.join("..", "figs", "q3_2_trainTest.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)   

        print("Best depth:", depths[np.argmin(val_errors)])     

    elif question == '4.1':

        dataset = utils.load_dataset('citiesSmall')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # part 1: implement knn.predict
        # part 2: print training and test errors for k=1,3,10 (use utils.classification_error)
        for k in [1,3,10]:
            model = KNN(k)
            model.fit(X,y)
            trainErr = np.mean(y != model.predict(X))
            testErr = np.mean(ytest != model.predict(Xtest))
            print('k={:2d}\ttrainErr: {:.3f}%\ttestErr: {:.3f}%'.format(k, trainErr, testErr))

        # part 3: plot classification boundaries for k=1 (use utils.plot_2dclassifier)
        model = KNN(1)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_1_knnDecisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '4.2':

        dataset = utils.load_dataset('citiesBig1')
        X = dataset['X']
        y = dataset['y']+1
        
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']+1

        # part 1: implement cnn.py
        # part 2: print training/test errors as well as number of examples for k=1
        model = KNN(1)
        model.fit(X, y)
        trainErr = np.mean(y != model.predict(X))
        testErr = np.mean(ytest != model.predict(Xtest))
        print('k=1\tnPts={}\ttrainErr: {:.3f}\ttestErr: {:.3f}'.format(model.X.shape[0], trainErr, testErr))

        # part 3: plot classification boundaries for k=1
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q4_2_cnnDecisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("n = %d, d = %d" % X.shape)
        
        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("Training error: %.3f" % tr_error)
            print("Testing error: %.3f" % te_error)
        
        print("Our implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))

        print("sklearn implementations")
        print("  Decision tree info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))    
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))

                
