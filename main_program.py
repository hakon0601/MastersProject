from run_config_settings import *
from NeuralNetwork.recurrent_neural_network import RecurrentNN
from NeuralNetwork.convolutional_neural_network import ConvolutionalNN


class MainProgram():
    def __init__(self):
        self.feature_extractor = None
        self.neural_network = None
        self.data_loader = None

    def extract_features(self, samples, samples_test):
        return self.feature_extractor.extract_features(samples), self.feature_extractor.extract_features(samples_test)

    def construct_neural_network(self, input_size=1000):
        self.neural_network.construct_neural_network(input_size=input_size)

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.neural_network.train_neural_network(samples, labels, samples_test, labels_test)

    def test_accuracy_of_solution(self, samples, labels, samples_test, labels_test):
        self.neural_network.test_accuracy_of_solution(samples, labels, samples_test, labels_test)

    def run(self, feature_extractor, neural_network, data_loader):
        self.feature_extractor = feature_extractor
        self.neural_network = neural_network
        self.data_loader = data_loader

        self.samples, self.labels, self.samples_test, self.labels_test = data_loader.load_data(recurrent=isinstance(self.neural_network, RecurrentNN))
        print(len(self.samples), "Samples loaded,", len(self.samples_test), "Test samples loaded")
        self.processed_samples, self.processed_samples_test = self.extract_features(self.samples, self.samples_test)
        self.feature_extractor.save_specgrams(samples=self.samples, labels=self.labels, select_random=True)
        self.construct_neural_network(input_size=len(self.processed_samples[0]))
        self.train_neural_network(self.processed_samples, self.labels, self.processed_samples_test, self.labels_test)
        self.test_accuracy_of_solution(self.processed_samples, self.labels, self.processed_samples_test, self.labels_test)
