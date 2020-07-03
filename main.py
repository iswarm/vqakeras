from config import config, loadDatasetConfig, parseArgs
import json
import os
import time
import tensorflow as tf
from preprocess import Preprocesser, bold, bcolored, writeline, writelist



'''
Trains /evaluates the model:
1. Set GPU configurations.
2. Preprocess data: reads from datasets, and convert into numpy arrays.
3. Builds the TF computational graph for the MAC model.
4. Starts a session and initialize / restores weights.
5. If config.train is True, trains the model for number of epochs:
    a. Trains the model on training data
    b. Evaluates the model on training / validation data, optionally with exponential-moving-average weights.
    c. Prints and logs statistics, and optionally saves model predictions.
    d. Optinally reduces learning rate if losses / accuracies don't improve, and applies early stopping.
6. If config.test is True, runs a final evaluation on the dataset and print final results!

'''
def main():
    with open(config.configFile(), "a+") as outFile:
        json.dump(vars(config), outFile)

    # set gpus
    if config.gpus != "":
        config.gpusNum = len(config.gpus.split(","))
        os.environ["CUDA_VISIBLE"] = config.gpus
    #tf.logging.set_verbosity(tf.logging.ERROR)

    # process data
    print(bold("Preprocess data..."))
    start = time.time()
    preprocessor = Preprocesser()
    data, embeddings, answeiDict = preprocessor.preprocessData()
    print("took {} seconds".format(bcolored("{:.2f}".format(time.time() - start), "blue")))




if __name__ == '__main__':
    parseArgs()
    loadDatasetConfig[config.dataset]()
    main()
