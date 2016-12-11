import sys
import tensorflow as tf
from NeuralNetwork.naural_network_base import NeuralNetworkBase
from run_config_settings import *
from math import floor
import numpy as np
import random
from tensorflow.python.ops import rnn, rnn_cell


class RecurrentNN(NeuralNetworkBase):
    def __init__(self, hidden_layers=[10], activation_functions_type=[0, 0], enable_bias=False, learning_rate=0.5, epocs=100):
        self.hidden_layers = hidden_layers
        self.activation_functions_type = activation_functions_type
        self.enable_bias = enable_bias
        self.learning_rate = learning_rate
        self.epocs = epocs



    def construct_neural_network(self, input_size=1000):
        self.input_size = int(input_size // RELATED_STEPS)
        output_size=NR_OF_CLASSES
        self.layers_size = [input_size] + self.hidden_layers + [output_size]

        state_size = 100
        num_layers = 3

        self.layer_tensors = []
        self.layer_tensors.append(tf.placeholder(tf.float32, [BATCH_SIZE, RELATED_STEPS]))

        self.layer_tensors.append(tf.placeholder(tf.float32, [BATCH_SIZE, RELATED_STEPS]))

        embeddings = tf.get_variable('embedding_matrix', [NR_OF_CLASSES, state_size])

        rnn_inputs = tf.nn.embedding_lookup(embeddings, self.layer_tensors[-1])

        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        init_state = cell.zero_state(BATCH_SIZE, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, NR_OF_CLASSES])
            b = tf.get_variable('b', [NR_OF_CLASSES], initializer=tf.constant_initializer(0.0))

        #reshape rnn_outputs and y so we can get the logits in a single matmul
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        y_reshaped = tf.reshape(self.layer_tensors[-1], [-1])

        logits = tf.matmul(rnn_outputs, W) + b

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
        self.train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

        self.init = tf.initialize_all_variables()

        return dict(
            x = self.layer_tensors[0],
            y = self.layer_tensors[-1],
            init_state = init_state,
            final_state = final_state,
            total_loss = self.cost,
            train_step = self.train_step
        )


        # Generate weights from input through hidden layers to output
        self.weights = []
        for i in range(1, len(self.layers_size) - 1):
            W = tf.Variable(tf.random_normal([self.layers_size[i], self.layers_size[i+1]]))
            self.weights.append(W)

        self.bias = []
        if self.enable_bias:
            for layer_size in self.layers_size[1:]:
                b = tf.Variable(tf.ones(([layer_size])))
                self.bias.append(b)

        self.keep_prob = tf.placeholder(tf.float32)

        # self.activation_model = self.model(x=self.layer_tensors[0])

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.activation_model, self.layer_tensors[-1]))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # self.predict_op = self.model(x=self.layer_tensors[0])

    def train_neural_network(self, samples, labels, samples_test, labels_test):
    #     tf.set_random_seed(2345)
    #     with tf.Session() as sess:
    #         sess.run(tf.initialize_all_variables())
    #         training_losses = []
    #         for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
    #             training_loss = 0
    #             steps = 0
    #             training_state = None
    #             for X, Y in epoch:
    #                 steps += 1
    #
    #                 feed_dict={g['x']: X, g['y']: Y}
    #                 if training_state is not None:
    #                     feed_dict[g['init_state']] = training_state
    #                 training_loss_, training_state, _ = sess.run([g['total_loss'],
    #                                                       g['final_state'],
    #                                                       g['train_step']],
    #                                                              feed_dict)
    #                 training_loss += training_loss_
    #             if verbose:
    #                 print("Average training loss for Epoch", idx, ":", training_loss/steps)
    #             training_losses.append(training_loss/steps)
    #
    #         if isinstance(save, str):
    #             g['saver'].save(sess, save)
    #
    # return training_losses



        samples = self.reshape_samples(samples)
        samples_test = self.reshape_samples(samples_test)
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.test_accuracy_of_solution(samples, labels, samples_test, labels_test, reshape=False)

        for epoch in range(self.epocs):
            avg_cost = 0.
            nr_of_batches_to_cover_all_samples = int(len(samples)/BATCH_SIZE)
            sys.stdout.write("\rTraining network %02d%%" % floor((epoch + 1) * (100 / self.epocs)))
            sys.stdout.flush()
            for j in range(nr_of_batches_to_cover_all_samples):
                batch_xs, batch_ys = self.get_random_batch(BATCH_SIZE, samples, labels)
                # Run optimization op (backprop)
                _, c = self.sess.run([self.train_step, self.cost], feed_dict={self.layer_tensors[0]: batch_xs, self.layer_tensors[-1]: batch_ys})

                avg_cost += c / nr_of_batches_to_cover_all_samples
            if epoch % 1 == 0:
                print("\tEpoch:", '%03d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
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
        index_of_correct_label = tf.argmax(self.layer_tensors[-1], 1)
        correct_predictions = tf.equal(index_of_highest_output_neurons, index_of_correct_label)
        # Computes the average of a list of booleans
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        accuracy_test = self.sess.run(accuracy, feed_dict={self.layer_tensors[0]: samples_test, self.layer_tensors[-1]: labels_test})
        accuracy_training = self.sess.run(accuracy, feed_dict={self.layer_tensors[0]: samples, self.layer_tensors[-1]: labels})
        print("Accuracy test:", accuracy_test, "Accuracy training:", accuracy_training)

    def model(self, x):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, self.input_size])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, RELATED_STEPS, x)

        lstm_cell = rnn_cell.BasicLSTMCell(self.layers_size[1], forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights[-1])
        #
        #
        # if len(self.layers_size) < 3:
        #     if self.enable_bias:
        #         return tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]
        #     else:
        #         return tf.matmul(self.layer_tensors[0], self.weights[0])
        # self.activations = []
        # if self.enable_bias:
        #     self.activations.append(tf.nn.relu(tf.matmul(self.layer_tensors[0], self.weights[0]) + self.bias[0]))
        # else:
        #     self.activations.append(tf.nn.relu(tf.matmul(self.layer_tensors[0], self.weights[0])))
        #
        # for i in range(1, len(self.weights) - 1):
        #     if self.enable_bias:
        #         self.activations.append(tf.nn.relu(tf.matmul(self.activations[i-1], self.weights[i]) + self.bias[i]))
        #     else:
        #         self.activations.append(tf.nn.relu(tf.matmul(self.activations[i-1], self.weights[i])))
        #
        # self.h_fc1_drop = tf.nn.dropout(self.activations[-1], self.keep_prob)
        # if self.enable_bias:
        #     return tf.matmul(self.h_fc1_drop, self.weights[-1] + self.bias[-1])
        # else:
        #     return tf.matmul(self.h_fc1_drop, self.weights[-1])


    def print_weights(self):
        print()
        for i in range(len(self.weights)):
            print("Weights layer: ", i)
            print(self.sess.run(self.weights[i]))
        if self.enable_bias:
            for j in range(len(self.bias)):
                print("Bias weights layer: ", j)
                print(self.sess.run(self.bias[j]))

    def predict_one_sample(self, sample):
        print(self.sess.run(self.predict_op, feed_dict={self.layer_tensors[0]: [sample]}))

    def reshape_samples(self, samples):
        # Cut of the rest of each sample after reshaping
        samples = np.array([sample[:self.input_size*RELATED_STEPS] for sample in samples])
        return samples.reshape((len(samples), RELATED_STEPS, self.input_size))

