import sys
import data
import model
import visualizer

DATA_SIZES = ['small', 'medium', 'large']

def main():
    if (len(sys.argv) > 0 and sys.argv[1] in DATA_SIZES):
        dataSize = sys.argv[1]
    else: 
        dataSize = DATA_SIZES[0]
    X_train, X_test, y_train, y_test = data.prepareData(dataSize)
    accuracy, trainedModel = model.evaluateAndTrainModel(X_train, X_test, y_train, y_test)
    print(f"Accuracy: {accuracy:.2%}")
    visualizer.visualizeData(trainedModel, X_test, y_test)
    
main()