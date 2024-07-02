# FileCleanService.py

from services.LoggingService import LoggingService
import os
import csv
import nltk  # Import the nltk library for natural language processing tasks
from nltk.corpus import stopwords  # Import the stopwords corpus for filtering out common words
from nltk.tokenize import word_tokenize  # Import the word_tokenize function for tokenizing text
from nltk.stem import WordNetLemmatizer  # Import the WordNetLemmatizer class for lemmatizing words
from pathlib import Path
import string
from models.cleanResults import cleanResults
import re #regex for string clean up 

class FileCleanService:
    def __init__(self, config, trainDataFnAndPath, testDataFnAndPath):
        self.logger = LoggingService(__name__).getLogger()
        self.config = config
        self.trainDataFnAndPath = trainDataFnAndPath
        self.testDataFnAndPath = testDataFnAndPath

    def __cleanUpFile(self, csvIn, csvOut):
        # Clean up the input file and write the cleaned data to the output file
        with open(csvOut, 'w', newline='', encoding="utf-8") as csvfileOut:
            csvwriter = csv.writer(csvfileOut, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            with open(csvIn, 'r', newline='', encoding="utf-8") as csvfileIn:
                csvreader = csv.reader(csvfileIn, delimiter=',', quotechar='"')
                for row in csvreader:
                    # Clean up the description column
                    description = self.clean_text(row[1])

                    # Write the cleaned row to the new file
                    csvwriter.writerow([row[0], description])

    def clean_text(self, text):
        # Clean the text by tokenizing, removing stopwords, lemmatizing, and removing punctuation
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

        # Define punctuation marks to remove
        punctuation_to_remove = ("'","`",'"',",","``","'s","--", "%")

        # Remove punctuation
        #cleaned_tokens = [token for token in lemmatized_tokens]
        cleaned_tokens = [token for token in lemmatized_tokens if token not in punctuation_to_remove]
        
        cleaned_text = ' '.join(cleaned_tokens)

        #Remove numbers. Doens't really change things.
        #cleaned_text = re.sub('\d', '', cleaned_text)
        
        return cleaned_text
    

    def cleanUpFiles(self):
        # Clean up the input files and return the paths of the cleaned files
        self.logger.info("Cleaning Input Files..")
        nltk.download('punkt') # Download the Punkt tokenizer for sentence tokenization
        nltk.download('stopwords') # Download the stopwords for filtering common words   

        # Set up clean data file paths to be the same as the input file paths
        trainDataCleanFnAndPath = os.path.join(os.path.dirname(self.trainDataFnAndPath), Path(self.trainDataFnAndPath).stem + "_clean" + Path(self.trainDataFnAndPath).suffix)
        testDataCleanFnAndPath = os.path.join(os.path.dirname(self.testDataFnAndPath), Path(self.testDataFnAndPath).stem + "_clean" + Path(self.testDataFnAndPath).suffix)

        # Clean up the train data file
        self.__cleanUpFile(self.trainDataFnAndPath, trainDataCleanFnAndPath)

        # Clean up the test data file
        self.__cleanUpFile(self.testDataFnAndPath, testDataCleanFnAndPath)

        return cleanResults(trainDataCleanFnAndPath, testDataCleanFnAndPath)