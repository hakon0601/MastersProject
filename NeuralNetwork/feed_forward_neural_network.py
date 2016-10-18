import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

class FeedForwardNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], enable_bias=False, learning_rate=0.5, training_iterations=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations


    def construct_neural_network(self, input_size=1000):
        output_size=NR_OF_CLASSES
        self.layers_size = [input_size] + self.hidden_layers + [output_size]
        self.layer_tensors = []
        # self.input_tensor = tf.placeholder(tf.float32, [None, input_size])

        # Creating a placeholder variable for keeping the values in each layer
        for layer_size in self.layers_size:
            self.layer_tensors.append(tf.placeholder(tf.float32, [None, layer_size]))


        # Generate weights from input through hidden layers to output
        self.weights = []
        for i in range(len(self.layers_size) - 1):
            W = tf.Variable(tf.ones([self.layers_size[i], self.layers_size[i+1]]))
            self.weights.append(W)

        # TODO fix bias support
        self.bias = []
        if self.enable_bias:
            for layer_size in self.layers_size[1:]:
                b = tf.Variable(tf.zeros(([layer_size])))
                self.bias.append(b)

        # Setting up activation functions between outgoing neurons and ongoing weights
        '''
        self.activation_functions = []
        for j in range(len(self.activation_functions_type)):
            if self.activation_functions_type[j] == 0:
               self.activation_functions.append(tf.nn.softmax(tf.matmul(self.layer_tensors[j], weights[j])))
        '''

        # self.activation_function = tf.nn.softmax(tf.matmul(self.input_tensor, W_1) + b)

        self.activation_model = self.model()

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation_model, self.layer_tensors[-1]))
        # self.cost = tf.reduce_mean(-tf.reduce_sum(self.layer_tensors[-1] * tf.log(self.activation_model), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        # self.predict_op = tf.argmax(self.activation_model, 1)
        self.predict_op = self.model()

        self.init = tf.initialize_all_variables()

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.print_weights()
        p = self.sess.run(self.activation_model, feed_dict={self.layer_tensors[0]: samples, self.layer_tensors[-1]: labels})
        g = self.sess.run(tf.nn.softmax_cross_entropy_with_logits(self.activation_model, self.layer_tensors[-1]), feed_dict={self.layer_tensors[0]: samples, self.layer_tensors[-1]: labels})

        for i in range(self.training_iterations):
            sys.stdout.write("\rTraining network %d%%" % floor((i + 1) * (100 / self.training_iterations)))
            sys.stdout.flush()

            batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
            self.sess.run(self.train_step, feed_dict={self.layer_tensors[0]: batch_xs, self.layer_tensors[-1]: batch_ys})
            # self.print_weights()
            # Test accuracy between each iteration to view improvement and stagnation
            self.test_accuracy_of_solution(samples_test, labels_test)
        print()
        self.print_weights()

    def get_next_batch(self, current_index, batch_size, samples, labels):
        current_index = current_index % len(samples)
        if current_index + batch_size < len(labels):
            return samples[current_index:current_index + batch_size], labels[current_index:current_index + batch_size]
        else:
            end = samples[current_index:], labels[current_index:]
            start = samples[:batch_size - len(end[0])], labels[:batch_size - len(end[1])]
            return end[0] + start[0], end[1] + start[1]

    def test_accuracy_of_solution(self, samples_test, labels_test):
        # print("\t Accuracy: ", np.mean(np.argmax(labels_test, axis=1) == self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test})))
        predictions_test = self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: samples_test})

        index_of_highest_output_neurons = tf.argmax(self.predict_op, 1)
        index_of_correct_label = tf.argmax(self.layer_tensors[-1], 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        # Computes the average of a list of booleans
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        print("\nAccuracy: ", self.sess.run(accuracy, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test}))

        # Predicting single samples one at the time
        for i in range(len(samples_test)):
            f = self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: [samples_test[i]]})
            d = [0, 1] if f[0][0] < f[0][1] else [1, 0]
            b = 1


    def model(self):
        if len(self.layers_size) < 3:
            if self.enable_bias:
                return tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]
            else:
                return tf.matmul(self.layer_tensors[0], self.weights[0])
        self.activations = []
        if self.enable_bias:
            self.activations.append(tf.nn.softmax(tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]))
        else:
            self.activations.append(tf.nn.softmax(tf.matmul(self.layer_tensors[0], self.weights[0])))

        for i in range(1, len(self.weights) - 1):
            if self.enable_bias:
                self.activations.append(tf.nn.softmax(tf.matmul(self.activations[i-1], self.weights[i]) + self.bias[i]))
            else:
                self.activations.append(tf.nn.softmax(tf.matmul(self.activations[i-1], self.weights[i])))

        if self.enable_bias:
            return tf.matmul(self.activations[-1], self.weights[-1] + self.bias[-1])
        else:
            return tf.matmul(self.activations[-1], self.weights[-1])


    def print_weights(self):
        print()
        for i in range(len(self.weights)):
            print("Weights layer: ", i)
            print(self.sess.run(self.weights[i]))
        if self.enable_bias:
            for j in range(len(self.bias)):
                print("Bias weights layer: ", j)
                print(self.sess.run(self.bias[j]))


'''
|1 0| * |1 1 1| = |1 1 1|
      * |1 1 1| =

|1 1 1| *   |1 1| = |3 3|
            |1 1|
            |1 1|

|0 1| * |1 1 1| = |1 1 1|
      * |1 1 1| =

|1 1 1| *   |1 1| = |3 3|
            |1 1|
            |1 1|
'''
