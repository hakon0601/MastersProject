import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

class FeedForwardNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], bias=False):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.bias = bias


    def construct_neural_network(self, input_size=1000):
        output_size=NR_OF_CLASSES
        self.layers_size = [input_size] + self.hidden_layers + [output_size]
        self.layer_tensors = []
        # self.input_tensor = tf.placeholder(tf.float32, [None, input_size])

        # Creating a placeholder variable for keeping the values in each layer
        for layer_size in self.layers_size:
            self.layer_tensors.append(tf.placeholder(tf.float32, [None, layer_size]))


        # Generate weights from input through hidden layers to output
        weights = []
        for i in range(len(self.layers_size) - 1):
            W = tf.Variable(tf.zeros([self.layers_size[i], self.hidden_layers[i+1]]))
            weights.append(W)

#        W_1 = tf.Variable(tf.zeros([input_size, output_size]))

        # TODO fix bias support
        if self.bias:
            b = tf.Variable(tf.zeros([output_size]))

        # Setting up activation functions between outgoing neurons and ongoing weights
        self.activation_functions = []
        for j in range(len(self.activation_functions_type)):
            if self.activation_functions_type[j] == 0:
               self.activation_functions.append(tf.nn.softmax(tf.matmul(self.layer_tensors[j], weights[j])))


        # self.activation_function = tf.nn.softmax(tf.matmul(self.input_tensor, W_1) + b)

        # self.output_tensor = tf.placeholder(tf.float32, [None, output_size]) # output layer size

        self.init = tf.initialize_all_variables()


    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for i in range(TRAINING_ITERATIONS):
            sys.stdout.write("\rTraining network %d%%" % floor((i + 1) * (100/TRAINING_ITERATIONS)))
            sys.stdout.flush()

            cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.output_tensor * tf.log(self.activation_function), reduction_indices=[1]))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

            batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
            self.sess.run(self.train_step, feed_dict={self.input_tensor: batch_xs, self.output_tensor: batch_ys})
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
        correct_prediction = tf.equal(tf.argmax(self.activation_function, 1), tf.argmax(self.output_tensor, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(self.sess.run(accuracy, feed_dict={self.input_tensor: samples_test, self.output_tensor: labels_test}))
