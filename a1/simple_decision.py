def predict(X):
    if X[1] > 37.669007:
        if X[0] > -96.090109:
            y = 1
        else:
            y = 2
    else: 
        if X[0] > -115.577574:
            y = 2
        else:
            y = 1
    return y
