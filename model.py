from sklearn.ensemble import RandomForestClassifier

def trainModel(X_train, y_train):
    model = RandomForestClassifier(n_estimators=142, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluateModel(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    return accuracy

def evaluateAndTrainModel(X_train, X_test, y_train, y_test):
    trainedModel = trainModel(X_train, y_train)
    accuracy = evaluateModel(trainedModel, X_test, y_test)
    return accuracy, trainedModel
