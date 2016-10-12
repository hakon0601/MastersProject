


import abc

class NeuralNetworkBase():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def construct_neural_network(self, samples, labels, samples_test, labels_test):
        return

    @abc.abstractmethod
    def train_neural_network(self, samples, labels, samples_test, labels_test):
        return

    @abc.abstractmethod
    def test_accuracy_of_solution(self, samples_test, labels_test):
        return
