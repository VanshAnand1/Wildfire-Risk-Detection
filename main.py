import data
import model
import visualizer

DATA_SIZES = ['small', 'medium', 'large']

def main():
    dataSize = DATA_SIZES[1]
    X_train, X_test, y_train, y_test = data.prepareData(dataSize)
    accuracy, trainedModel = model.evaluateAndTrainModel(X_train, X_test, y_train, y_test)
    print(f"Accuracy: {accuracy:.2%}")
    visualizer.visualizeData(trainedModel, X_test, y_test)
    
main()