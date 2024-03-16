import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_accuracy = 0

    def early_stop(self, val_accuracy):
        if val_accuracy > self.max_accuracy:
            self.max_accuracy = val_accuracy
            self.counter = 0
        elif val_accuracy < (self.max_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
