import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import utils
from kmeans import Kmeans
from kmedians import Kmedians
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN
import linear_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '1.1', '1.2', '1.3', '1.4', '2', '2.2', '4', '4.1', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        X = utils.load_dataset('clusterData')['X']

        model = Kmeans(k=4)
        model.fit(X)
        utils.plot_2dclustering(X, model.predict(X))
        
        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.1':
        X = utils.load_dataset('clusterData')['X']

        # part 1: implement kmeans.error
        # part 2: get clustering with lowest error out of 50 random initialization

        best_model = None
        min_error = np.inf
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            error = model.error(X)
            if error < min_error:
                min_error = error
                best_model = model

        utils.plot_2dclustering(X, best_model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_50_inits.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.2':

        # part 3: plot min error across 50 random inits, as k is varied from 1 to 10
        X = utils.load_dataset('clusterData')['X']

        minErrs = []
        for k in range(1,11):
            best_model = None
            min_error = np.inf
            for i in range(50):
                model = Kmeans(k)
                model.fit(X)
                error = model.error(X)
                if error < min_error:
                    min_error = error
                    best_model = model

            minErrs.append(min_error)

        plt.plot(list(range(1,11)), minErrs)
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.title('k-means training error as k increases')

        fname = os.path.join("..", "figs", "kmeans_err_k.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.3':

        X = utils.load_dataset('clusterData2')['X']


        def closure_1_3_1():
            k = 4
            best_model = None
            min_error = np.inf
            for i in range(50):
                model = Kmeans(k)
                model.fit(X)
                error = model.error(X)
                if error < min_error:
                    min_error = error
                    best_model = model

            plt.figure()
            utils.plot_2dclustering(X, best_model.predict(X))

            fname = os.path.join("..", "figs", "kmeans_outliers_best_model.png")
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)

        def closure_1_3_2():
            minErrs = []
            for k in range(1,11):
                best_model = None
                min_error = np.inf
                for i in range(50):
                    model = Kmeans(k)
                    model.fit(X)
                    error = model.error(X)
                    if error < min_error:
                        min_error = error
                        best_model = model

                minErrs.append(min_error)

            plt.figure()
            plt.plot(list(range(1,11)), minErrs)
            plt.xlabel('k')
            plt.ylabel('Error')
            plt.title('k-means training error as k increases')

            fname = os.path.join("..", "figs", "kmeans_err_k_outliers.png")
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)

        # part 3: implement kmedians.py
        # part 4: plot kmedians.error

        # def closure_1_3_4():
        #     minErrs = []
        #     for k in range(1,11):
        #         best_model = None
        #         min_error = np.inf
        #         for i in range(50):
        #             model = kmedians.fit(X, k)
        #             error = model['error'](model,X)
        #             if error < min_error:
        #                 min_error = error
        #                 best_model = model

        #         minErrs.append(min_error)

        #     plt.figure()
        #     plt.plot(list(range(1,11)), minErrs)
        #     plt.xlabel('k')
        #     plt.ylabel('Error')
        #     plt.title('k-medians training error as k increases')

        #     fname = os.path.join("..", "figs", "q3_3_kmedians_err_k_outliers.png")
        #     plt.savefig(fname)
        #     print("\nFigure saved as '%s'" % fname)

        def closure_1_3_4():
            k = 4
            best_model = None
            min_error = np.inf
            for i in range(50):
                model = Kmedians(k)
                model.fit(X)
                error = model.error(X)
                if error < min_error:
                    min_error = error
                    best_model = model


            plt.figure()
            utils.plot_2dclustering(X, best_model.predict(X))

            fname = os.path.join("..", "figs", "kmedians_outliers_best_model.png")
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)


        closure_1_3_1()
        closure_1_3_2()
        closure_1_3_4()

    if question == '1.4':
        X = utils.load_dataset('clusterData2')['X']
        
        model = DBSCAN(eps=1, min_samples=3)
        y = model.fit_predict(X)

        utils.plot_2dclustering(X,y)

        fname = os.path.join("..", "figs", "clusterdata_dbscan.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '2':
        img = utils.load_dataset('dog')['I']/255

        # part 1: implement quantize_image.py
        # part 2: use it on the doge
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img, b)
            d_img = quantizer.dequantize(q_img)

            fname = os.path.join("..", "figs", "doge_{}.png".format(b))
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)

    elif question == "4":
            # loads the data in the form of dictionary
            data = utils.load_dataset("outliersData")
            X = data['X']
            y = data['y']

            # Plot data
            plt.figure()
            plt.plot(X,y,'b.',label = "Training data")
            plt.title("Training data")

            # Fit least-squares estimator
            model = linear_model.LeastSquares()
            model.fit(X,y)
            print(model.w)

            # Draw model prediction
            Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
            yhat = model.predict(Xsample)
            plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
            plt.legend()
            figname = os.path.join("..","figs","least_squares_outliers.pdf")
            print("Saving", figname)
            plt.savefig(figname)

    elif question == "4.1":
            data = utils.load_dataset("outliersData")
            X = data['X']
            y = data['y']

            ''' YOUR CODE HERE '''
            # Fit weighted least-squares estimator
            z = np.concatenate(([1]*400,[0.1]*100),axis = 0)
            model = linear_model.WeightedLeastSquares()
            model.fit(X,y,z)

            # Draw model prediction
            Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
            yhat = model.predict(Xsample)
            plt.figure()
            plt.plot(X,y,'b.',label = "Training data")
            plt.title("Training data")
            plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
            plt.legend()
            figname = os.path.join("..","figs","least_squares_outliers_weighted.pdf")
            print("Saving", figname)
            plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)
