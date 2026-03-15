from sklearn.ensemble import RandomForestClassifier

def trainModel(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluateModel(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy

def evaluateAndTrainModel(X_train, X_test, y_train, y_test):
    model = trainModel(X_train, y_train)
    accuracy = evaluateModel(model, X_test, y_test)
    return accuracy
