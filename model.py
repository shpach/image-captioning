import tensorflow as tf
from pretrained.vgg16 import vgg16

import time
import numpy as np
from pretrained.imagenet_classes import class_names



class ImageCaptioner(object):
    def __init__(self, config, word_table):

        self.config = config
        self.word_table = word_table
        # Create session
        self.session = tf.Session()

        # Create architecture
        self.imgs_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.cnn_output = None
        self.build_cnn()
        #self.build_rnn()

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep = 100)
        
        # load shared weights if necessary
        if config.cnn_model_file:
            self.cnn.load_weights(config.cnn_model_file, self.session)



    def build_cnn(self):
        print('Building CNN...')

        if self.config.cnn_model == 'custom':
            self.build_custom_cnn()

        else:
            self.build_vgg16()

    def build_custom_cnn(self):
        print('Building custom model...')

        W_conv1 = _weight_variable([5, 5, 1, 32])
        b_conv1 = _bias_variable([32])
        h_conv1 = tf.nn.relu(_conv2d(imgs_placeholder, W_conv1) + b_conv1)
        h_pool1 = _max_pool_2x2(h_conv1)

        W_conv2 = _weight_variable([5, 5, 32, 64])
        b_conv2 = _bias_variable([64])
        h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = _max_pool_2x2(h_conv2)

        W_conv3 = _weight_variable([5, 5, 64, 128])
        b_conv3 = _bias_variable([128])
        h_conv3 = tf.nn.relu(_conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = _max_pool_2x2(h_conv3)
        h_flat3 = tf.reshape(h_pool3, [-1])

        # TODO: Possibly add some FC layers here

        self.cnn_output = h_flat3

    def build_vgg16(self):
        print('Building VGG-16...')
        self.cnn = vgg16(self.imgs_placeholder, sess=self.session, trainable=self.config.train_cnn)
        # self.cnn_output = conv_net.fc2


    def build_rnn(self):
        print('Building RNN...')

        # contexts = conv_feats
        # feats = fc_feats
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        vector_dim = self.config.vector_dim
        learning_rate = self.config.learning_rate
        num_words = self.word_table.num_words
        max_num_words = self.config.max_word_len
        vector_dim = self.config.vector_dim

        # Inputs to RNN
        self.rnn_input = tf.placeholder(tf.float32, [batch_size, vector_dim])
        self.sentences = tf.placeholder(tf.int32, [batch_size, max_num_words])
        self.mask = tf.placeholder(tf.int32, [batch_size, max_num_words])

        # Outputs of RNN
        gen_captions = []
        
        lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)      
        state = [tf.zeros([batch_size, s]) for s in lstm.state_size]

        W_word = tf.Variable(tf.random_uniform([hidden_size, num_words]))
        b_word = tf.Variable(tf.zeros([num_words]))

        total_loss = 0.0

        for idx in range(max_num_words):
            if idx == 0:
                curr_emb = self.rnn_input
            else:
                func_idx2words = np.vectorize(self.word_table.idx2word.get)
                curr_emb = tf.nn.embedding_lookup(self.word_table.word2vec, func_idx2words(self.sentences[:,idx-1]))
                    
            output, state = lstm(curr_emb, state)

            logits = tf.matmul(output, W_word)+b_word

            ####################################################
            # XXX: Might want another FC layer afterwards here #
            ####################################################

            # Generate captions
            max_prob_word = tf.argmax(logits, 1)
            gen_captions.append(max_prob_word)

            onehot_labels = tf.cast(self.sentences[:,idx], dtype=tf.int32)
            logits = tf.cast(logits, dtype=tf.float32)

            # Compute loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)*self.mask[:,idx]
            loss = tf.reduce_sum(cross_entropy)
            total_loss = total_loss + loss
            # NOTE: Might need to use "tf.get_variable_scope().reuse_variables()"
            
        self.gen_captions = tf.stack(gen_captions, axis=1)
        self.total_loss = total_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        
    def train(self, data):

        print("Training network...")
        start_time = time.time()
        
        word2idx = self.word_table.word2idx
        idx2word = self.word_table.idx2word
        train_images = data.training_data
        train_caps = data.training_annotation

        max_word_len = self.config.max_word_len
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        display_loss = self.config.display_loss
        
        train_idx = np.arange(len(train_caps))
        
        shuffled_train_images = np.zeros(len(train_images))
        shuffled_train_caps = {}
        
        batch_num = 0
        for epoch in range(num_epochs):
            # shuffle training data
            np.random.shuffle(train_idx)
            for idx, old_idx in enumerate(train_idx):
                shuffled_train_images[idx] = train_images[old_idx]
                shuffled_train_caps[idx] = train_caps[old_idx]
            
            for batch_idx in range(0,len(train_caps),batch_size):
                curr_image = shuffled_train_images[batch_idx:batch_idx+batch_size]
                curr_caps = shuffled_train_caps[batch_idx:batch_idx+batch_size]
                
                curr_sentences = np.zeros((len(batch_size),max_word_len))
                curr_mask = np.zeros((len(batch_size),max_word_len))
                
                for cap_idx, cap in enumerate(curr_caps):
                    for word_idx, word in enumerate(cap.lower().split(' ')[:-1]):
                        curr_sentences[cap_idx][word_idx] = word2idx[word]
                        curr_mask[cap_idx][word_idx] = 1
                         
                if self.config.train_cnn:
                    print('Not implemented yet!')

                else: 
                    conv_output = self.session.run(self.conv_output, feed_dict={self.imgs_placeholder: curr_image})
                    _, total_loss = self.session.run([self.train_op, self.total_loss], feed_dict={
                        self.rnn_input : conv_output, 
                        self.sentences : curr_sentences,
                        self.mask : curr_mask
                        })
                
                if batch_num%display_loss == 0:
                    pass
                    #print("Current Training Loss = " + str(total_loss))
                        
                batch_num += 1

        print("Finished Training")
        print("Elapsed time: ", self.elapsed(time.time() - start_time))
            
    def elapsed(sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/(60*60)) + " hr"

 

    def test(self, data):
        """ Test the model. """
        print("Testing model...")
        result_file = self.config.results_file

        test_images = data.validation_data
        test_caps = data.validation_annotation

        max_word_len = self.config.max_word_len

        captions = []


        for img in test_images:
            if self.config.train_cnn:
                print('Not implemented yet!')

            else:
                conv_output = self.session.run(self.conv_output, feed_dict={self.imgs_placeholder: img})
                img_cap = self.session.run(self.gen_captions, feed_dict={
                        self.rnn_input : conv_output, 
                        self.sentences : curr_sentences,
                        self.mask : curr_mask
                        })
                captions.append(img_cap)

        print(captions)
        
        
    # Layers/initializers
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

    def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def _max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
