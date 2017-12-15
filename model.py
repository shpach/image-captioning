import tensorflow as tf
from pretrained.vgg16 import vgg16

import time
import numpy as np
from pretrained.imagenet_classes import class_names




class ImageCaptioner(object):
    def __init__(self, config, word_table):

        self.config = config
        self.word_table = word_table
        self.training_flag = True
        # Create session
        self.session = tf.Session()

        # Create architecture
        self.imgs_placeholder = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.build_cnn()
        self.build_rnn()

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep = 100)
		
        checkpoint = tf.train.get_checkpoint_state(config.ckpt_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

        self.train_writer  = tf.summary.FileWriter(config.summary_file)
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
        self.cnn_output = self.cnn.fc2
        self.img_dim = 4096


    def build_rnn(self):
        print('Building RNN...')

        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        vector_dim = self.config.vector_dim
        learning_rate = self.config.learning_rate
        num_words = self.word_table.num_words
        max_num_words = self.config.max_word_len
        vector_dim = self.config.vector_dim

        # Inputs to RNN
        self.rnn_input = tf.placeholder(tf.float32, [None, self.img_dim])
        self.sentences = tf.placeholder(tf.int32, [None, max_num_words])
        self.mask = tf.placeholder(tf.float32, [None, max_num_words])

        # Outputs of RNN
        gen_captions = []
        
        # Squeeze conv output dimensions to RNN input dimensions
        W_conv2rnn = _weight_variable([self.img_dim, vector_dim])
        b_conv2rnn = _bias_variable([vector_dim]) 
        fc_conv2rnn = tf.nn.xw_plus_b(self.rnn_input, W_conv2rnn, b_conv2rnn)
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        state = lstm_cell.zero_state(batch_size, dtype=tf.float32)	#[tf.zeros([batch_size, s]) for s in lstm.state_size]

        idx2vec_np = np.array([self.word_table.word2vec[self.word_table.idx2word[i]] for i in range(num_words) if self.word_table.idx2word[i] in self.word_table.word2vec])
        self.idx2vec = tf.convert_to_tensor(idx2vec_np, dtype=tf.float32)

        W_word = tf.Variable(tf.random_uniform([hidden_size, num_words]))
        b_word = tf.Variable(tf.zeros([num_words]))

        total_loss = 0.0
        

        for idx in range(max_num_words):
            if idx == 0:
                curr_emb = fc_conv2rnn
            else:
                if self.training_flag:
                    curr_emb = tf.nn.embedding_lookup(self.idx2vec, self.sentences[:, idx-1])
                else:
                    curr_emb = tf.nn.embedding_lookup(self.idx2vec, max_prob_word)

            if self.config.num_lstm == 1:
                output, state = lstm_cell(curr_emb, state)
            else:
                output, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=curr_emb, dtype=tf.float32)

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

        self.total_loss = total_loss / tf.reduce_sum(self.mask)
        tf.summary.scalar('total_loss',self.total_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
			
        self.merged = tf.summary.merge_all()

        
    def train(self, data):
        print("Training network...")
        start_time = time.time()
        self.training_flag = True
        
        word2idx = self.word_table.word2idx
        idx2word = self.word_table.idx2word
        train_images = data.training_data
        train_caps = data.training_annotation

        max_word_len = self.config.max_word_len
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        display_loss = self.config.display_loss
        
        train_idx = np.arange(len(train_caps))

        batch_num = 0
        for epoch in range(num_epochs):
            print("Epoch number: ", epoch)
            # shuffle training data
            shuffled_train_images = []
            shuffled_train_caps = []
            np.random.shuffle(train_idx)
            for old_idx in train_idx:
                shuffled_train_images.append(train_images[old_idx])
                shuffled_train_caps.append(train_caps[old_idx])
            
            for batch_idx in range(0,len(train_caps),batch_size):
                curr_images = shuffled_train_images[batch_idx:batch_idx+batch_size]
                curr_caps = shuffled_train_caps[batch_idx:batch_idx+batch_size]
                
                curr_sentences = np.zeros((batch_size,max_word_len))
                curr_mask = np.zeros((batch_size,max_word_len))
                for cap_idx, cap in enumerate(curr_caps):
                    for word_idx, word in enumerate(cap.lower().split(' ')[:-1]):
                        if word_idx == max_word_len:
                            break
                        if word in word2idx:
                            curr_sentences[cap_idx][word_idx] = word2idx[word]
                        else:
                            curr_sentences[cap_idx][word_idx] = word2idx["<RARE>"]
                        curr_mask[cap_idx][word_idx] = 1

                    if word_idx != max_word_len:
                        curr_sentences[cap_idx][word_idx+1] = word2idx["<END>"]
                        curr_mask[cap_idx][word_idx+1] = 1
			
                #curr_mask = np.ones((batch_size, max_word_len))
            
                if self.config.train_cnn:
                    pass
                    #print('Not implemented yet!')

                else: 

                    cnn_output = self.session.run(self.cnn_output, feed_dict={self.imgs_placeholder: curr_images})

                    _, summary, total_loss = self.session.run([self.train_op, self.merged, self.total_loss], feed_dict={
                                                self.rnn_input : cnn_output,
                                                self.sentences : curr_sentences,
                                                self.mask : curr_mask,
                                                })
                
                if batch_num%display_loss == 0:
                    print("Current Training Loss = " + str(total_loss))
                    self.train_writer.add_summary(summary, batch_num)
	
				
                        
                batch_num += 1

            if epoch%self.config.ckpt_freq == 0:
                self.saver.save(self.session, self.config.ckpt_dir+'Captioner', global_step=epoch)

        print("Finished Training")
        print("Elapsed time: ", self.elapsed(time.time() - start_time))
            
    def elapsed(self,sec):
        if sec<60:
            return str(sec) + " sec"
        elif sec<(60*60):
            return str(sec/60) + " min"
        else:
            return str(sec/(60*60)) + " hr"

 

    def test(self, data):
        """ Test the model. """
        print("Testing model...")
        self.training_flag = False
        results_file = self.config.results_file
        max_num_words = self.config.max_word_len

        test_images = data.training_data
        test_caps = data.training_annotation

        max_word_len = self.config.max_word_len

        captions = []
        
        # create empty matrices fed for testing
        empty_sentences = np.zeros((len(test_images), max_num_words))
        empty_mask = np.ones((len(test_images), max_num_words))

        if self.config.train_cnn:
            print('Not implemented yet!')

        else:
            cnn_output = self.session.run(self.cnn_output, feed_dict={self.imgs_placeholder: test_images})
            print(cnn_output.shape)
            print('Convolutional features computed.')
            captions_idx = self.session.run(self.gen_captions, feed_dict={
                                        self.rnn_input : cnn_output,
                                        self.sentences : empty_sentences,
                                        self.mask : empty_mask,
                                        })
        
        captions = []
        for x in range(len(captions_idx)):
            captions.append([])
            for y in range(len(captions_idx[0])):
                captions[x].append(self.word_table.idx2word[captions_idx[x][y]])
					
        output_text = ""
        for cap_idx in range(len(captions)):
            if "<END>" in captions[cap_idx]:
                end_sentence = np.argmax(np.array(captions[cap_idx])=="<END>")
            else:
                end_sentence = max_num_words
            sentence = ' '.join(captions[cap_idx][:end_sentence])
            output_text += sentence + "\n"

        f = open(results_file, "w")
        f.write(output_text)
        f.close()

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
