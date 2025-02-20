import os
import kaggle

class Dataset:
    def __init__(self, dataset_name, output_type):
        self.dataset_name = dataset_name
        self.output_type = output_type
        self.data = self.load_data()

    def load_data(self):
        # Download the dataset from Kaggle
        kaggle.api.dataset_download_files(self.dataset_name, path='./path of the testing datasets', unzip=True)

        # Load the data from the downloaded files
        pass

    def process_data(self):
        # Process the data based on the output type
        pass