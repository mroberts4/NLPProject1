# main.py
# Use config.json to toggle models and vectorizer combos on or off
# Import necessary modules and classes
from services.LoggingService import LoggingService
from services.ConfigService import ConfigService
from services.FileCleanService import FileCleanService
from services.TrainModelService import TrainModelService
from services.TestModelService import TestModelService
import os
import pyfiglet
import nltk
from csv import DictWriter
import time
from datetime import timedelta, datetime

import json
from types import SimpleNamespace

# Download the required NLTK data
nltk.download('wordnet')

class main:
    def __init__(self):
        # Initialize the main class
        self.logger = LoggingService(__name__).getLogger()
        self.config = ConfigService().config
        self.logger.info(f"\n\n{pyfiglet.figlet_format(self.config['appName'])}By The Word Wizards\nTexas A&M University - Corpus Christi\nCOSC 6590 - Selected Topics: Natural Language Processing\nProject 1 - Summer 2023\n\n")

        # Set the input directory and file paths
        self.inputDir = os.path.join(os.getcwd(), "_input")
        self.outputDir = os.path.join(os.getcwd(), "_output")
        trainDataFnAndPath = os.path.join(self.inputDir, "trainData.csv")
        testDataFnAndPath = os.path.join(self.inputDir, "testData.csv")
        self.outputResults = os.path.join(self.outputDir, "results.csv")

        # Initialize the required services
        self.FileCleanServ = FileCleanService(self.config, trainDataFnAndPath, testDataFnAndPath)
        self.TrainServ = TrainModelService(self.config, self.outputDir)
        self.TestServ = TestModelService(self.config)

        self.logger.info(f"App init complete...")

    def run(self):
        # Execute the main program logic
        self.logger.info(f"App running...")

        # Clean up files (only need to do this once)
        cleanResults = self.FileCleanServ.cleanUpFiles()

        for testParam in self.config['testParams']:
            testParem = json.dumps(testParam)
            testParamObj = json.loads(testParem, object_hook=lambda d: SimpleNamespace(**d))
            if not testParamObj.enabled:
                continue

            # about to train and test so set the start time
            start_time = time.time()

            # Train the model
            trainResults = self.TrainServ.trainModel(cleanResults.trainFnAndPath, modelName=testParamObj.model, vectorName=testParamObj.vector)

            # Run the model on test data
            testResults = self.TestServ.testModel(trainResults, cleanResults.testFnAndPath)

            # done training and testing so set the end time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Output the results
            # Print the vectorizer and model information
            self.logger.info(f"Using Vectorizer: {trainResults.vectorizerName}, Using Model: {trainResults.modelName}")
            # to console and log file
            self.logger.info(f"Training accuracy: {trainResults.accuracy:.4f}, elapsed: {trainResults.elapsedTime}")
            self.logger.info(f"Testing accuracy: {testResults.accuracy:.4f}, elapsed: {testResults.elapsedTime}")
            self.logger.info(f"Confusion Matrix: \n{testResults.confusion_matrix}")  # Display the confusion matrix
            
            self.logger.info(f"")

            # to csv file
            outputResultsFieldNames = ['Timestamp', 'Model', 'Vectorizer', 'TrainElapsed', 'TrainAccuracy', 'TestElapsed', 'TestAccuracy', 'TotalElapsed']
            outputResultsExists = os.path.exists(self.outputResults)
            with open(self.outputResults, 'a', newline='') as f_object:
                dictwriter_object = DictWriter(f_object, fieldnames=outputResultsFieldNames, delimiter=',', quotechar='"')
                dict = {'Timestamp':datetime.now().strftime("%m/%d/%Y %H:%M:%S"), 
                        'Model':trainResults.modelName, 
                        'Vectorizer':trainResults.vectorizerName, 
                        'TrainElapsed':str(timedelta(seconds=trainResults.elapsedTime)), 
                        'TrainAccuracy': "%.4f" % trainResults.accuracy, 
                        'TestElapsed':str(timedelta(seconds=testResults.elapsedTime)), 
                        'TestAccuracy':"%.4f" % testResults.accuracy,
                        'TotalElapsed':str(timedelta(seconds=elapsed_time))}
                
                if not outputResultsExists:
                    dictwriter_object.writeheader()

                # write append details
                dictwriter_object.writerow(dict)
                f_object.close()


# Create an instance of the main class and run the program
myApp = main()
myApp.run()
