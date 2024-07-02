# TestModelService.py

# Import necessary modules and classes
from services.LoggingService import LoggingService
import csv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer  # Import the TfidfVectorizer class for converting text data into numerical features using TF-IDF transformation
from sklearn.linear_model import LogisticRegression  # Import the LogisticRegression class for training logistic regression models
from sklearn.pipeline import Pipeline  # Import the Pipeline class for constructing a pipeline of data transformation steps followed by an estimator
from sklearn.metrics import accuracy_score, confusion_matrix  # Import the accuracy_score function for calculating the accuracy of model predictions
from models.testResults import testResults
import time

class TestModelService:
    def __init__(self, config):
        # Initialize TestModelService class
        self.logger = LoggingService(__name__).getLogger()
        self.config = config
        self.confusion_matrices = {}  # Initialize confusion_matrices attribute

    def loadModel(self, modelPath, vectorizerPath):
        # Load the trained model and vectorizer from file
        trained_model = joblib.load(modelPath)
        vectorizer = joblib.load(vectorizerPath)
        return trained_model, vectorizer

    def testModel(self, trainResults, testCleanFnAndPath):
        # Test the trained model on the test data
        start_time = time.time()
        self.logger.info("Testing Model..")

        # Load the trained model and vectorizer
        trained_model, vectorizer = self.loadModel(trainResults.modelPathAndFn, trainResults.vectorizerPathAndFn)

        # Read the test data from the CSV file
        texts, labels = self.readData(testCleanFnAndPath)

        # Predict the sentiment using the trained model
        sentiment = trained_model.predict(texts)

        # Calculate accuracy on test data
        accuracy = accuracy_score(labels, sentiment)

        # Get the confusion matrix for the current model/vectorizer combo run
        confusion_matrix_for_run = confusion_matrix(labels, sentiment)

        # Format and print the results
        """
        result_str = f"\nUsing Vectorizer: {trainResults.vectorizerName}, Using Model: {trainResults.modelName}\n"
        result_str += f"Training accuracy: {trainResults.accuracy:.2%}, Testing accuracy: {accuracy:.2%}\n"
        result_str += "Confusion Matrix:\n"
        result_str += str(confusion_matrix_for_run)
        print(result_str)
        """
        end_time = time.time()
        elapsed_time = end_time - start_time

        return testResults(accuracy=accuracy, elapsed_time=elapsed_time, confusion_matrix=confusion_matrix_for_run)
    
    def readData(self, filename):
        # Read data from the CSV file and return texts and labels
        texts = []
        labels = []
        with open(filename, "r", newline='', encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=',', quotechar='"')
            for row in reader:
                texts.append(row["description"])
                labels.append(row["Category"])
        return texts, labels