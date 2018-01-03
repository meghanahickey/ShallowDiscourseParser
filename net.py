import tensorflow as tf
import numpy as np
import json
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from pprint import pprint


class NeuralNet:

    def __init__(self):

        self.feature_length = 0
        training, self.vocab, self.labels = self._read_data('train')
        self.vocab_size = len(self.vocab)
        print('vocab size:', self.vocab_size)
        self.dev = self._read_data('dev')
        self.test = self._read_data('test')
        self.num_labels = len(self.labels)
        print('num labels:', self.num_labels)

        # used to obtain features for summed graph model
        self.train_features, self.train_labels = self._summed_bow(training, self.labels)
        # changing the input to this method from dev to test, so that it can be passed to summed graph
        self.dev_features, self.dev_labels = self._summed_bow(self.dev, self.labels)
        self.dev_num = len(self.dev_features)
        self.train_num = len(self.train_features)
        self.num_dimensions = len(self.train_features[0])
        self.learning_rate = 0.01

    def create_output(self):
        print('classifying!')

        predicted_labels = self._run_summed_graph()
        label_dict = {i: self.labels[i] for i in range(self.num_labels)}
        with open('output_dev.json', 'w+') as output_file:
            for i, instance in enumerate(self.test):

                if len(instance['Connective']['TokenList']) == 0:
                    connective = {'TokenList': []}
                else:
                    connective = {'TokenList': [instance['Connective']['TokenList'][j][2]
                                                for j in range(len(instance['Connective']['TokenList']))]}

                label = label_dict[np.argmax(predicted_labels[i])]
                entry = {'Sense': [label], 'DocID': instance['DocID'], 'Type': instance['Type'],
                         'Connective': connective,
                         'Arg1': {'TokenList': [word[2] for word in instance['Arg1']['TokenList']]},
                         'Arg2': {'TokenList': [word[2] for word in instance['Arg2']['TokenList']]}}
                json.dump(entry, output_file)
                output_file.write('\n')

    def _read_data(self, stage):
        data = []
        labels = set()
        vocab = {}
        with open(stage + '/relations.json', 'r') as input_file:
            for line in input_file:
                entry_dict = json.loads(line)
                words = entry_dict['Arg1']['RawText'].split() + \
                        entry_dict['Arg2']['RawText'].split() + \
                        entry_dict['Connective']['RawText'].split()
                for word in words:
                    vocab[word] = vocab.get(word, 0) + 1
                labels.add(entry_dict['Sense'][0])
                data.append(entry_dict)

        if stage is not 'train':
            return data

        shortened_vocab = [word for word in vocab if vocab[word] > 1.0]
        shortened_vocab.append('<UNK>')
        vocab = {word: shortened_vocab.index(word) for word in shortened_vocab}
        return data, vocab, list(labels)

    def _summed_bow(self, instances, labels):
        print('creating bag of words features using summation as pooling function')
        features_vectors = []
        label_vectors = []
        self._load_vectors()
        vocab = set(self.vectors.keys())
        zeros = np.array([0] * 50)
        for instance in instances:
            features = []
            features.extend(sum([self.vectors[word.lower()] if word.lower() in vocab else zeros
                                 for word in self._tokenize(instance['Arg1']['RawText'])]))
            features.extend(sum([self.vectors[word.lower()] if word.lower() in vocab else zeros
                                 for word in self._tokenize(instance['Arg2']['RawText'])]))
            if len(instance['Connective']['RawText'].split()) > 0:
                features.extend(sum([self.vectors[word.lower()] if word.lower() in vocab else zeros
                                     for word in word_tokenize(instance['Connective']['RawText'])]))
            else:
                features.extend(zeros)
            label_vectors.append([1 if instance['Sense'] == label else 0 for label in labels])
            features_vectors.append(np.array(features))
        return features_vectors, label_vectors

    def _load_vectors(self):

        # assign all numbers the same random vector
        vector_dict = {"<NUM>": np.random.rand(50)}

        with open('glove.6B.50d.txt') as input_file:
            for line in input_file:
                tokens = line.split()
                vector_dict[tokens[0]] = np.array([float(token) for token in tokens[1:]])
        self.vectors = vector_dict

    def _tokenize(self, string):
        """this method takes care of string preprocessing"""

        string = re.sub(r"(\d+[.,]?)+", "<NUM>", string)

        string = string.replace('-', ' ')
        string = string.replace('.', ' ')
        string = string.replace('`', ' ')
        string = string.replace("'", " ")
        return word_tokenize(string)

    def _basic_bow(self, instances, vocab, labels):
        feature_vectors = []
        label_vectors = []
        instance_count = 0
        max_instance_length = 0
        for instance in instances:

            instance_count += 1
            text = Counter(instance['Arg1']['RawText'].split()
                           + instance['Arg2']['RawText'].split()
                           + instance['Connective']['RawText'].split())
            if len(text) > max_instance_length:
                max_instance_length = len(text)
            feature_vector = [vocab[word] if word in vocab.keys() else vocab['<UNK>'] for word in text]
            label_vector = [1 if instance['Sense'] == label else 0 for label in labels]
            feature_vectors.append(feature_vector)
            label_vectors.append(label_vector)

        print('max instance length:', max_instance_length)
        if self.feature_length != 0:
            feature_length = self.feature_length
        else:
            feature_length = max_instance_length
            self.feature_length = max_instance_length

        padded_feature_vectors = []
        for instance in feature_vectors:
            if len(instance) < feature_length:
                new_instance = instance + [0] * (feature_length - len(instance))
                padded_feature_vectors.append(new_instance)
            else:
                padded_feature_vectors.append(instance)
        padded_feature_vectors = np.array(padded_feature_vectors)

        return padded_feature_vectors[:, :feature_length], label_vectors

    def _bag_of_words(self, instances, labels):
        print('creating bag of words features')
        feature_vectors = []
        label_vectors = []

        max_arg_length = 25
        max_conn_length = 2
        self._load_vectors()
        vectors = self.vectors
        zeros = np.array([0] * 50)
        vocab = set(vectors.keys())
        for instance in instances:
            arg1_vector = []
            arg1_words = self._tokenize(instance['Arg1']['RawText'])
            arg1_num = len(arg1_words)
            conn_vector = []
            conn_words = self._tokenize(instance['Connective']['RawText'])
            conn_num = len(conn_words)
            arg2_vector = []
            arg2_words = self._tokenize(instance['Arg2']['RawText'])
            arg2_num = len(arg2_words)
            for i in range(max_arg_length):
                if i < arg1_num:
                    arg1_vector.append(vectors[arg1_words[i]] if arg1_words[i] in vocab else zeros)
                else:
                    arg1_vector.append(zeros)
                if i < arg2_num:
                    arg2_vector.append(vectors[arg2_words[i]] if arg2_words[i] in vocab else zeros)
                else:
                    arg2_vector.append(zeros)

            for i in range(max_conn_length):
                if i < conn_num:
                    conn_vector.append(vectors[conn_words[i]] if conn_words[i] in vocab else zeros)
                else:
                    conn_vector.append(zeros)

            label_vector = np.array([1 if instance['Sense'] == label else 0 for label in labels])
            feature_vectors.append(np.array(arg1_vector + conn_vector + arg2_vector))
            label_vectors.append(label_vector)

        return feature_vectors, label_vectors

    def _run_summed_graph(self):
        hidden_size = 25
        num_epochs = 1
        output_size = self.num_labels
        batch_size = 100

        input = tf.placeholder(tf.float32, (None, self.num_dimensions))

        predicted = tf.nn.softmax(tf.contrib.layers.fully_connected(
            tf.contrib.layers.fully_connected(input, hidden_size),
            output_size))

        gold_y = tf.placeholder(tf.float32, [batch_size, output_size])
        cross_entropy = - tf.reduce_sum(gold_y * tf.log(tf.clip_by_value(predicted, 1e-10, 1.0)), axis=1)

        learning_rate = self.learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print('training time')
        for epoch in range(num_epochs):
            for i in range(int(self.train_num/ batch_size)):
                print('iteration', i)
                sess.run(
                    train_step,
                    feed_dict={
                        input: self.train_features[i * batch_size:(i + 1) * batch_size],
                        gold_y: self.train_labels[i * batch_size:(i + 1) * batch_size]})

        results = [sess.run([predicted], feed_dict={input: feature.reshape((1,150))})[0]
                   for feature in self.dev_features]
        print(len(results))
        return results

    def _run_basic_graph(self, test_features):
        hidden_size = 64
        num_iters = 100
        output_size = self.num_labels
        batch_size = self.dev_num

        # embeddings = tf.Variable(tf.random_uniform((vocab_size, char_embed_size), -1, 1))

        W1 = tf.Variable(tf.random_uniform((100, hidden_size), -1, 1))
        b1 = tf.Variable(tf.zeros((1, hidden_size)))
        W2 = tf.Variable(tf.random_uniform((hidden_size, output_size), -1, 1))
        b2 = tf.Variable(tf.zeros((1, output_size)))

        x = tf.placeholder(tf.float32, (self.dev_num, 100))

        predicted = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(x, W1) + b1), W2) + b2)

        train_features = self.train_features
        train_labels = self.train_labels

        print('initialize first session')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        prelim_results = sess.run([predicted], feed_dict={x: test_features[:batch_size]})
        print('prelim results:', prelim_results)

        gold_y = tf.placeholder(tf.float32, [self.dev_num, output_size])
        cross_entropy = - tf.reduce_sum(gold_y * tf.log(tf.clip_by_value(predicted, 1e-10, 1.0)), axis=1)

        learning_rate = self.learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print('training time')
        for iteration in range(num_iters):
            # import pdb; pdb.set_trace()
            for i in range(int(len(train_features) / batch_size)):
                sess.run(
                    train_step,
                    feed_dict={
                        x: train_features[i * batch_size:(i + 1) * batch_size],
                        gold_y: train_labels[i * batch_size:(i + 1) * batch_size]})
        results = []
        print('test features size:', test_features.shape)
        for i in range(int(len(test_features) / batch_size)):
            result_list = sess.run([predicted], feed_dict={x: test_features[i * batch_size:(i + 1) * batch_size]})[0]
            for j in range(batch_size):
                results.append(result_list[j])
        print(len(results))
        return results

def main():
    nn = NeuralNet()
    nn.create_output()


if __name__ == '__main__':
    main()





