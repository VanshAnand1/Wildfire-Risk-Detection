import data
import model

def main():
    dataSize = 'large'
    X_train, X_test, y_train, y_test = data.prepareData(dataSize)
    accuracy = model.evaluateAndTrainModel(X_train, X_test, y_train, y_test)
    
main()