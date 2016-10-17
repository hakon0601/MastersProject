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
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], bias=False, learning_rate=0.5, training_iterations=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.bias = bias
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


#        W_1 = tf.Variable(tf.zeros([input_size, output_size]))

        # TODO fix bias support
        if self.bias:
            b = tf.Variable(tf.zeros([output_size]))

        # Setting up activation functions between outgoing neurons and ongoing weights
        '''
        self.activation_functions = []
        for j in range(len(self.activation_functions_type)):
            if self.activation_functions_type[j] == 0:
               self.activation_functions.append(tf.nn.softmax(tf.matmul(self.layer_tensors[j], weights[j])))
        '''

        # self.activation_function = tf.nn.softmax(tf.matmul(self.input_tensor, W_1) + b)

        # self.output_tensor = tf.placeholder(tf.float32, [None, output_size]) # output layer size

        self.activation_model = self.model()

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation_model, self.layer_tensors[-1]))
        cost = tf.reduce_mean(-tf.reduce_sum(self.layer_tensors[-1] * tf.log(self.activation_model), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        # self.predict_op = tf.argmax(self.activation_model, 1)
        self.predict_op = self.model()

        self.init = tf.initialize_all_variables()


    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)
        #self.print_weights()

        for i in range(self.training_iterations):
            sys.stdout.write("\rTraining network %d%%" % floor((i + 1) * (100 / self.training_iterations)))
            sys.stdout.flush()

            batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
            self.sess.run(self.train_step, feed_dict={self.layer_tensors[0]: batch_xs, self.layer_tensors[-1]: batch_ys})
            self.print_weights()
            # Test accuracy between each iteration to view improvement and stagnation
            self.test_accuracy_of_solution(samples_test, labels_test)
        print()
        # correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        #
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(self.sess.run(accuracy, feed_dict={self.x: samples_test, self.y_: labels_test}))

    def get_next_batch(self, current_index, batch_size, samples, labels):
        if current_index + batch_size < len(labels):
            return samples[current_index:current_index + batch_size], labels[current_index:current_index + batch_size]
        else:
            end = samples[current_index:], labels[current_index:]
            start = samples[:batch_size - len(end[0])], labels[:batch_size - len(end[1])]
            return end[0] + start[0], end[1] + start[1]

    def test_accuracy_of_solution(self, samples_test, labels_test):
        print("\t Accuracy: ", np.mean(np.argmax(labels_test, axis=1) == self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test})))
        a = self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test})
        c = np.argmax(labels_test, axis=1)
        correct_prediction = tf.equal(tf.argmax(self.activation_model, 1), tf.argmax(self.layer_tensors[-1], 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy old: ", self.sess.run(accuracy, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test}))
        predictions = self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: samples_test})
        for i in range(len(samples_test)):
            f = self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: [samples_test[i]]})
            a = self.sess.run(self.weights)
            b = 1


    def model(self):
        # TODO remove, just for testing
        return tf.matmul(tf.matmul(self.layer_tensors[0], self.weights[0]), self.weights[1])
        h = tf.nn.sigmoid(tf.matmul(self.layer_tensors[0], self.weights[0]))
        for i in range(1, len(self.weights) - 1):
            h = tf.nn.sigmoid(tf.matmul(h, self.weights[i]))
        return tf.matmul(h, self.weights[-1])

    def print_weights(self):
        print()
        print(self.sess.run(self.weights[0]))
        print()
        print(self.sess.run(self.weights[1]))


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
