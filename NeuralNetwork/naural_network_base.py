


import abc

class NeuralNetworkBase():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def construct_neural_network(self, samples, labels, samples_test, labels_test):
        return