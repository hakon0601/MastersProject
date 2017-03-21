import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import numpy as np
import random


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

class FeedForwardNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], enable_bias=False, learning_rate=0.5, dropout_rate=0.9, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate


    def construct_neural_network(self, input_size=1000):
        output_size=NR_OF_CLASSES
        self.layers_size = [input_size] + self.hidden_layers + [output_size]
        self.layer_tensors = []
        # self.input_tensor = tf.placeholder(tf.float32, [None, input_size])

        # Creating a placeholder variable for keeping the values in each layer
        for layer_size in self.layers_size:
            self.layer_tensors.append(tf.placeholder(tf.float32, [None, layer_size]))
        print("Network structure", self.layers_size)


        # Generate weights from input through hidden layers to output
        self.weights = []
        for i in range(len(self.layers_size) - 1):
            W = tf.Variable(tf.random_normal([self.layers_size[i], self.layers_size[i+1]]))
            self.weights.append(W)

        self.bias = []
        if self.enable_bias:
            for layer_size in self.layers_size[1:]:
                b = tf.Variable(tf.random_normal(([layer_size])))
                self.bias.append(b)

        self.keep_prob = tf.placeholder(tf.float32)

        self.activation_model = self.model()

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation_model, self.layer_tensors[-1]))
        # self.cost = tf.reduce_mean(-tf.reduce_sum(self.layer_tensors[-1] * tf.log(self.activation_model), reduction_indices=[1]))
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.predict_op = self.model()

        index_of_highest_output_neurons = tf.argmax(self.activation_model, 1)
        index_of_correct_label = tf.argmax(self.layer_tensors[-1], 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        # Computes the average of a list of booleans
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        tf.scalar_summary("cost", self.cost)
        tf.scalar_summary("accuracy", self.accuracy)

        self.summary_op = tf.merge_all_summaries()

        self.init = tf.global_variables_initializer()

    def train_neural_network(self, samples, labels, samples_test, labels_test):
        self.sess = tf.Session()
        self.sess.run(self.init)
        # self.test_accuracy_of_solution(samples, labels, samples_test, labels_test)

        writer = tf.train.SummaryWriter(LOG_PATH, graph=tf.get_default_graph())

        for epoch in range(self.epochs):
            avg_cost = 0.
            nr_of_batches_to_cover_all_samples = int(len(samples)/BATCH_SIZE)
            sys.stdout.write("\rTraining network %02d%%" % floor((epoch + 1) * (100 / self.epochs)))
            sys.stdout.flush()
            for j in range(nr_of_batches_to_cover_all_samples):
                # batch_xs, batch_ys = self.get_next_batch(i*BATCH_SIZE, BATCH_SIZE, samples, labels)
                batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)
                _, summary = self.sess.run([self.train_step, self.summary_op], feed_dict={self.layer_tensors[0]: batch_xs, self.layer_tensors[-1]: batch_ys, self.keep_prob: self.dropout_rate})
                writer.add_summary(summary, epoch * nr_of_batches_to_cover_all_samples + j)

            if epoch % 1 == 0:
                self.test_accuracy_of_solution(samples, labels, samples_test, labels_test)
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

    def test_accuracy_of_solution(self, samples, labels, samples_test, labels_test):
        accuracy_test = self.sess.run(self.accuracy, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test, self.keep_prob: 1})
        accuracy_training = self.sess.run(self.accuracy, feed_dict={self.layer_tensors[0]: samples, self.layer_tensors[-1]: labels, self.keep_prob: 1})
        print("Accuracy test:", accuracy_test, "Accuracy training:", accuracy_training)

    def model(self):
        if len(self.layers_size) < 3:
            if self.enable_bias:
                return tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]
            else:
                return tf.matmul(self.layer_tensors[0], self.weights[0])
        self.activations = []
        if self.enable_bias:
            self.activations.append(tf.nn.relu(tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]))
        else:
            self.activations.append(tf.nn.relu(tf.matmul(self.layer_tensors[0], self.weights[0])))

        for i in range(1, len(self.weights) - 1):
            if self.enable_bias:
                self.activations.append(tf.nn.relu(tf.matmul(self.activations[i-1], self.weights[i]) + self.bias[i]))
            else:
                self.activations.append(tf.nn.relu(tf.matmul(self.activations[i-1], self.weights[i])))

        self.h_fc1_drop = tf.nn.dropout(self.activations[-1], self.keep_prob)
        if self.enable_bias:
            return tf.matmul(self.h_fc1_drop, self.weights[-1]) + self.bias[-1]
        else:
            return tf.matmul(self.h_fc1_drop, self.weights[-1])


    def print_weights(self):
        print()
        for i in range(len(self.weights)):
            print("Weights layer: ", i)
            print(self.sess.run(self.weights[i]))
        if self.enable_bias:
            for j in range(len(self.bias)):
                print("Bias weights layer: ", j)
                print(self.sess.run(self.bias[j]))