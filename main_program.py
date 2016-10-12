

class MainProgram():

    def __init__(self, feature_extractor, neural_network, data_loader):
        self.feature_extractor = feature_extractor
        self.neural_network = neural_network
        self.data_loader = data_loader

        self.samples, self.labels, self.samples_test, self.labels_test = data_loader.load_data()
        self.processed_samples, self.processed_samples_test = self.extract_features(self.samples, self.samples_test)
        self.construct_neural_network(self.processed_samples, self.labels, self.processed_samples_test, self.labels_test)
        self.train_neural_network(self.processed_samples, self.labels, self.processed_samples_test, self.labels_test)
        self.test_accuracy_of_solution(self.processed_samples_test, self.labels_test)

    def extract_features(self, samples, samples_test):
        return self.feature_extractor.extract_features(samples), self.feature_extractor.extract_features(samples_test)

    def construct_neural_network(self, samples, labels, samples_test, labels_test):
        self.neural_network.construct_neural_network(nr_of_hidden_layers=)

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.neural_network.train_neural_network(samples, labels, samples_test, labels_test)

    def test_accuracy_of_solution(self, samples_test, labels_test):
        self.neural_network.test_accuracy_of_solution(samples_test, labels_test)