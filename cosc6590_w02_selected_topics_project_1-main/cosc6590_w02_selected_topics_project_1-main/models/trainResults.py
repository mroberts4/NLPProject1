# trainResults.py
class trainResults:
    def __init__(self, model_name=None, vectorizer_name=None, model_pathAndFn=None, vectorizer_pathAndFn=None, elapsed_time=None, accuracy=None):
        self.modelPathAndFn = model_pathAndFn
        self.vectorizerPathAndFn = vectorizer_pathAndFn
        self.elapsedTime = elapsed_time
        self.accuracy = accuracy  # Corrected the attribute name to 'accuracy'
        self.modelName = model_name
        self.vectorizerName = vectorizer_name
