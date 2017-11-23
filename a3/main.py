import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.1":
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        ''' YOUR CODE HERE'''
        # Fit the least squares model with bias
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y) ** 2) / n
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest) ** 2) / t
        print("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X, y, 'b.', label="Training Data")
        plt.title('Training Data with bias')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X), np.max(X), 1000)[:, None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat, yhat, 'g', label="Least Squares with bias prediction")
        plt.legend()
        figname = os.path.join("..", "figs", "Least_Squares_with_bias.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.2":

        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        for p in range(11):
            print("p=%d" % p)

            ''' YOUR CODE HERE '''
            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            # Compute training error
            yhat = model.predict(X)
            trainError = np.sum((yhat - y) ** 2) / n
            print("Training error = %.0f" % trainError)

            # Compute test error

            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest) ** 2) / t
            print("Test error     = %.0f" % testError)

            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))
            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]


            '''YOUR CODE HERE'''
            #Predict on Xhat
            yhat = model.predict(Xhat)
            plt.plot(Xhat, yhat, 'g', label="Least Squares with basis {}".format(p))

            plt.legend()
            figname = os.path.join("..","figs","PolyBasis%d.pdf"%p)
            print("Saving", figname)
            plt.savefig(figname)

    elif question == "2.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.sum((yhat - yvalid)**2)/ (n//2)
            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "2.2":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            all_data = np.hstack((X, y))
            np.random.shuffle(all_data)
            # Perform Cross Validation
            fold_length = all_data.shape[0]//10
            sum_errors = 0
            for i in range(10):
                valid_data = all_data[i*fold_length:(i+1)*fold_length][:]
                train_data = np.vstack((all_data[0:i*fold_length], all_data[(i+1)*fold_length:all_data.shape[0]]))

                Xtrain = train_data[:,0]
                Xtrain= np.reshape(Xtrain, (Xtrain.shape[0],1))
                ytrain = train_data[:,1]

                Xvalid = valid_data[:,0]
                Xvalid = np.reshape(Xvalid, (Xvalid.shape[0],1))
                yvalid = valid_data[:,1]


                # Train on the training set
                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(Xtrain,ytrain)

                # Compute the error on the validation set
                yhat = model.predict(Xvalid)
                validError = np.sum((yhat - yvalid)**2)/ (n//2)
                sum_errors += validError
                print("Error with sigma = {:e} = {}".format( sigma ,validError))

            validError = sum_errors / 10
            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf_cross_val.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # RESULT:
        #  TRAINING ERROR: 0.000
        #  VALIDATION ERROR: 0.084
        #  NonZeros: 101

        # Note:
        #  This is different than the MATLAB solution which
        #  achieved a validation error of 0.082

    elif question == "3.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # RESULT:
        #  TRAINING ERROR: 0.002
        #  VALIDATION ERROR: 0.074
        #  NonZeros: 101

    elif question == "3.2":
        # Load Binary and Multi -class data
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL1 model
        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # RESULT:
        #  TRAINING ERROR: 0.000
        #  VALIDATION ERROR: 0.052
        #  NonZeros: 71

    elif question == "3.3":
        # Load Binary and Multi -class data
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL0 model
        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # RESULT:
        #  TRAINING ERROR: 0.000
        #  VALIDATION ERROR: 0.040
        #  NonZeros: 24

        # NOTE:
        #  The Matlab version selects feature 2 instead of 44
        #  at the 24th epoch, which results in validation error
        #  of 0.018 instead of 0.040. I suspect that this is due to the different
        #  floating-point precision that Matlab and Python use.
        #  Python, by default, uses higher floating point precision.


    elif question == "4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Run Q3 given example - Fit One-vs-all Least Squares
        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        # RESULT:
        #  TRAINING ERROR: 0.160
        #  VALIDATION ERROR: 0.134

    elif question == "4.1":
        # Load Binary and Multi -class data
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit One-vs-all Logistic Regression
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        # RESULT:
        #  TRAINING ERROR: 0.084
        #  VALIDATION ERROR: 0.070

    elif question == "4.4":
        # Load Binary and Multi -class data
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        # RESULT:
        #  TRAINING ERROR: 0.000
        #  VALIDATION ERROR: 0.008

        # NOTE:
        #  Although the validation error is different than in the A4 sol file,
        #  it's the same result achieved by Mark's Matlab code. I believe Mark made
        #  a typo in the solution file.
