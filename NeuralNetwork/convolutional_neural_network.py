import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import numpy as np
import random
from tensorflow.python.ops import rnn, rnn_cell


class ConvolutionalNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], enable_bias=False, learning_rate=0.5, dropout_rate=0.9, cell_type=0 , time_related_steps=20, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.time_related_steps = time_related_steps
        self.epochs = epochs

    def construct_neural_network(self, input_size=1000):
        tf.reset_default_graph()
        input_size = int(input_size // self.time_related_steps)
        self.layers_size = [input_size] + self.hidden_layers + [NR_OF_CLASSES]

        self.input_tensor = tf.placeholder(tf.float32, [None, self.time_related_steps, self.layers_size[0]])
        self.output_tensor = tf.placeholder(tf.float32, [None, self.layers_size[-1]])
        self.keep_prob = tf.placeholder(tf.float32)

        self.activation_model = self.model()
        self.cost = -tf.reduce_sum(self.output_tensor * tf.log(self.activation_model))
        # self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        tf.get_variable_scope().reuse_variables()
        samples = self.reshape_samples(samples)
        samples_test = self.reshape_samples(samples_test)
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)

        for epoch in range(self.epochs):
            nr_of_batches_to_cover_all_samples = int(len(samples)/BATCH_SIZE)
            sys.stdout.write("\rTraining network %02d%%\t" % floor((epoch + 1) * (100 / self.epochs)))
            sys.stdout.flush()
            for j in range(nr_of_batches_to_cover_all_samples):
                batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)
                self.sess.run(self.train_step, feed_dict={self.input_tensor: batch_xs, self.output_tensor: batch_ys, self.keep_prob: self.dropout_rate})

            self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)
        print("Optimization Finished!")

    def get_next_batch(self, current_index, batch_size, samples, labels):
        current_index = current_index % len(samples)
        if current_index + batch_size < len(labels):
            return samples[current_index:current_index + batch_size], labels[current_index:current_index + batch_size]
        else:
            end = samples[current_index:], labels[current_index:]
            start = samples[:batch_size - len(end[0])], labels[:batch_size - len(end[1])]
            return end[0] + start[0], end[1] + start[1]

    def get_random_batch(self, batch_size, samples, labels):
        rand_samples = []
        rand_labels = []
        for i in range(batch_size):
            rand_index = random.randrange(0, len(samples))
            rand_samples.append(samples[rand_index])
            rand_labels.append(labels[rand_index])
        return rand_samples, rand_labels

    def test_accuracy_of_solution(self, samples, labels, samples_test, labels_test, reshape=True):
        if reshape:
            samples = self.reshape_samples(samples)
            samples_test = self.reshape_samples(samples_test)

        index_of_highest_output_neurons = tf.argmax(self.activation_model, 1)
        index_of_correct_label = tf.argmax(self.output_tensor, 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        accuracy_test = self.sess.run(accuracy, {self.input_tensor: samples_test, self.output_tensor: labels_test, self.keep_prob: 1})
        accuracy_training = self.sess.run(accuracy, {self.input_tensor: samples, self.output_tensor: labels, self.keep_prob: 1})
        print("Accuracy test:", accuracy_test, "Accuracy training:", accuracy_training)

    def model(self):
        cells = []
        for i in range(1, len(self.layers_size) - 1):
            if self.cell_type == 0:
                cell = rnn_cell.BasicLSTMCell(self.layers_size[i])
            elif self.cell_type == 1:
                cell = rnn_cell.BasicRNNCell(self.layers_size[i])
            elif self.cell_type == 2:
                cell = rnn_cell.GRUCell(self.layers_size[i])
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cells.append(cell)
        multilayer_cell = rnn_cell.MultiRNNCell(cells)
        multilayer_cell = rnn_cell.DropoutWrapper(multilayer_cell, output_keep_prob=self.keep_prob)

        output, state = tf.nn.dynamic_rnn(multilayer_cell, self.input_tensor, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1) # TODO this may be a bottleneck (memory)

        last_weights = tf.Variable(tf.random_normal([self.layers_size[-2], self.layers_size[-1]]))
        if self.enable_bias:
            bias = tf.Variable(tf.random_normal(([self.layers_size[-1]])))
            return tf.nn.softmax(tf.matmul(last, last_weights) + bias)
        return tf.nn.softmax(tf.matmul(last, last_weights))

    def reshape_samples(self, samples):
        # Cut of the rest of each sample after reshaping
        samples = np.array([sample[:self.layers_size[0]*self.time_related_steps] for sample in samples])
        return samples.reshape((len(samples), self.time_related_steps, self.layers_size[0]))