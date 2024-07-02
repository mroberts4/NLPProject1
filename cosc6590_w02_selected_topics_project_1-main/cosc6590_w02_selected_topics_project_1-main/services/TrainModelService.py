# TrainModelService.py

from services.LoggingService import LoggingService
import os
import csv
import joblib  # Import the joblib module for saving and loading Python objects to and from files
from sklearn.feature_extraction.text import TfidfVectorizer  # Import the TfidfVectorizer class for converting text data into numerical features using TF-IDF transformation
from sklearn.feature_extraction.text import HashingVectorizer  # Import the HashingVectorizer class for converting text data into numerical features using hashing
from sklearn.feature_extraction.text import CountVectorizer  # Import the CountVectorizer class for converting text data into numerical features by counting word occurrences
from sklearn.linear_model import LogisticRegression  # Import the LogisticRegression class for training logistic regression models
from sklearn.ensemble import RandomForestClassifier  # Import the RandomForestClassifier class for training random forest models
from sklearn.svm import SVC  # Import the SVC class for training support vector classifiers
from sklearn.naive_bayes import MultinomialNB  # Import the MultinomialNB class for training multinomial Naive Bayes models
from sklearn.pipeline import Pipeline  # Import the Pipeline class for constructing a pipeline of data transformation steps followed by an estimator
from sklearn.metrics import accuracy_score  # Import the accuracy_score function for calculating the accuracy of model predictions
from sklearn.preprocessing import MinMaxScaler #fixed import
from sklearn.preprocessing import StandardScaler
from models.trainResults import trainResults
import time
from datetime import datetime

class TrainModelService:
    def __init__(self, config, outputDir):
        # Initialize the TrainModelService
        self.logger = LoggingService(__name__).getLogger()
        self.config = config
        self.outputDir = outputDir

    def trainModel(self, trainCleanFnAndPath, modelName="MultinomialNB", vectorName="CountVectorizer"):
        # Train the specified model using the specified vectorizer
        start_time = time.time()
        self.logger.info("Training Model..")

        # Read the training data from the CSV file
        texts, labels = self.readData(trainCleanFnAndPath)

        # Initialize the Vectorizer
        # Options: TfidfVectorizer, CountVectorizer, HashingVectorizer
        # Vectorizer converts text data into numerical features.
        # Each option has different characteristics for feature extraction.
        if vectorName.lower() == "countvectorizer":
            self.vectorizerName = "CountVectorizer"
            # Parameters for CountVectorizer:
            # - stop_words: Specify a set of stop words to remove during tokenization.
            # - ngram_range: Tuple (min_n, max_n) defining the range of n-values for word n-grams.
            #   For example, (1, 2) means unigrams and bigrams will be considered.
            #   Default is (1, 1) for unigrams only.
            # self.vectorizer = CountVectorizer()
            # Other Example:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))
            # self.vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
            # self.vectorizer = CountVectorizer(stop_)
        elif vectorName.lower() == "tfidfvectorizer":
            self.vectorizerName = "TfidfVectorizer"
            # Parameters for TfidfVectorizer:
            # - stop_words: Specify a set of stop words to remove during tokenization.
            # - ngram_range: Tuple (min_n, max_n) defining the range of n-values for word n-grams.
            #   For example, (1, 2) means unigrams and bigrams will be considered.
            #   Default is (1, 1) for unigrams only.
            # - use_idf: Enable inverse-document-frequency reweighting.
            self.vectorizer = TfidfVectorizer()
            # Other Example:
            # self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), use_idf=False)
            # self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), use_idf=True)
        elif vectorName.lower() == "hashingvectorizer":
            self.vectorizerName = "HashingVectorizer"
            # Parameters for HashingVectorizer:
            # - alternate_sign: If True, the output of the HashingVectorizer will have alternating signs.
            #   This is useful for certain models like MultinomialNB, which expects positive feature values.
            # - n_features: The number of features (dimensions) in the output vectors.
            #   Default is 2^20. A larger value can potentially improve accuracy at the cost of memory.
            if modelName.lower() == "multinomialnb":
                self.vectorizer = HashingVectorizer(alternate_sign=False)
                # Other Example:                    
                # self.vectorizer = HashingVectorizer(alternate_sign=False, n_features=2**19)
            else:
                self.vectorizer = HashingVectorizer()
                # Other Example:
                # self.vectorizer = HashingVectorizer(n_features=2**19)
        else:
            self.vectorizerName = "CountVectorizer"
            self.vectorizer = CountVectorizer()


        # Initialize the classification model
        # Options: LogisticRegression, RandomForestClassifier, SVC, MultinomialNB
        # Different models have various approaches to learning and prediction.
        if modelName.lower() == "logisticregression":
            self.modelName = "LogisticRegression"
            self.model = LogisticRegression(max_iter=300)
            # Hyperparameter options for Logistic Regression:
            # - C: Inverse of regularization strength. Smaller values specify stronger regularization.
            # - max_iter: Maximum number of iterations for the solver to converge.
            # - solver: Algorithm to use in the optimization problem. 
            #   Options include 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'.
            # Other Examples:
            # self.model = LogisticRegression(solver='liblinear', max_iter=2000, C=0.1)
            # self.model = LogisticRegression(solver='sag', max_iter=1000)
            
            
        elif modelName.lower() == "randomforestclassifier":
            self.modelName = "RandomForestClassifier"
            self.model = RandomForestClassifier()
            # Hyperparameter options for Random Forest Classifier:
            # - n_estimators: Number of trees in the forest.
            # - max_depth: Maximum depth of the tree.
            # Other Example:
            # self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
            
        elif modelName.lower() == "svc":
            self.modelName = "SVC"
            self.model = SVC()
            # Hyperparameter options for Support Vector Classifier (SVC):
            # - kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.).
            # - C: Regularization parameter. Larger values specify less regularization.
            # Other Example:
            # self.model = SVC(kernel='linear', C=1.0)
            
        elif modelName.lower() == "multinomialnb":
            self.modelName = "MultinomialNB"
            self.model = MultinomialNB()
            # Hyperparameter options for Multinomial Naive Bayes (MultinomialNB):
            # - alpha: Additive (Laplace/Lidstone) smoothing parameter. 
            #   Larger values specify stronger smoothing.
            # Other Example:
            # self.model = MultinomialNB(alpha=0.1)
            
        else:
            self.modelName = "LogisticRegression"
            self.model = LogisticRegression()


        # Create a pipeline with the vectorizer and the model
        pipeline = Pipeline([("vectorizer", self.vectorizer), ("model", self.model)])
        # Fit the pipeline to the training data
        pipeline.fit(texts, labels)

        # Calculate accuracy on training data
        train_predictions = pipeline.predict(texts)
        training_accuracy = accuracy_score(labels, train_predictions)

        # Save the trained model and vectorizer to files
        fnPrefix = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_pathAndFn = os.path.join(self.outputDir, f"{fnPrefix}_trained_model_{self.modelName}.pkl")
        vectorizer_pathAndFn = os.path.join(self.outputDir, f"{fnPrefix}_vectorizer_{self.vectorizerName}.pkl")
        trainedModel = joblib.dump(pipeline, model_pathAndFn)
        trainedVectorizer = joblib.dump(self.vectorizer, vectorizer_pathAndFn)

        self.logger.info("Model training complete.")
        end_time = time.time()
        elapsed_time = end_time - start_time

       # Create a trainResults object to store the training results
        return trainResults(model_name=self.modelName, 
                            vectorizer_name=self.vectorizerName, 
                            model_pathAndFn=model_pathAndFn, 
                            vectorizer_pathAndFn=vectorizer_pathAndFn, 
                            elapsed_time=elapsed_time, 
                            accuracy=training_accuracy)  # Assign training_accuracy to accuracy


    def readData(self, filename):
        # Read the training data from a CSV file.
        # Each row contains 'description' and 'Category'.
        texts = []
        labels = []
        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=',', quotechar='"')
            try:
                for row in reader:
                    texts.append(row["description"])
                    labels.append(row["Category"])
            except:
                self.logger.exception(f"Error with line_num {reader.line_num}", exc_info=False)
                raise
        return texts, labels