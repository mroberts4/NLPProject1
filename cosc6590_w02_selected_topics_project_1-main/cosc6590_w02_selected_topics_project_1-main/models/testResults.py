# testResults.py
class testResults:
    def __init__(self, accuracy, elapsed_time, confusion_matrix=None):
        self.accuracy = accuracy
        self.elapsed_time = elapsed_time
        self.confusion_matrix = confusion_matrix
        self.testAccuracy = accuracy
        self.elapsedTime = elapsed_time  # Add this line to store the elapsed time during testing
