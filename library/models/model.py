from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
import torch

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, audio_file):
        # Extract features from the audio file using the model
        inputs = self.tokenizer(audio_file, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def test(self, dataset):
        # Test the model on the dataset
        predictions = []
        for audio_file in dataset.data:
            prediction = self.__call__(audio_file)
            predictions.append(prediction)
        return predictions

    def evaluate(self, y_true, y_pred):
        # Calculate performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        return accuracy, precision, recall, f1

def get_model(model_name):
    return Model(model_name)