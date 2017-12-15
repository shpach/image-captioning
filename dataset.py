import os
import numpy as np
import pickle
import wordtable
import cv2

class DataSet():
    def __init__(self, batch_size, is_training, width, height, save_file):
        self.training_data = None
        self.validation_data = None
        self.training_annotation = []
        self.validation_annotation = []
        self.batch_size = batch_size
        self.is_training = is_training
        self.image_width = width
        self.image_height = height
        self.save_file = save_file

    def setup(self, dir, num2train, word_table):
        """" build the training dataset """
        training_file = os.path.join(dir, 'Flickr8k_text/Flickr_8k.trainImages.txt')
        with open(training_file, encoding="utf8") as f:
            image_list = []
            im_idx = 0
            for line in f:
                file_name = line.strip()
                img = cv2.imread(dir + 'Flickr8k_image/' + file_name)
                img = resize_image(img, self.image_width, self.image_height)
                image_list.append(img)
                self.training_annotation.append([])
                for image_id in range(5):
                    self.training_annotation[im_idx].append(word_table.img2sentence[file_name + '#' + str(image_id)])
                if im_idx >= num2train-1:
                    break
                im_idx += 1
            self.training_data = np.asarray(image_list)


        count = 0
        """" build the validation dataset """
        validate_file = os.path.join(dir, 'Flickr8k_text/Flickr_8k.devImages.txt')
        with open(validate_file, encoding="utf8") as f:
            image_list = []
            for line in f:
                count += 1
                file_name = line.strip()
                img = cv2.imread(dir + 'Flickr8k_image/' + file_name)
                img = resize_image(img, self.image_width, self.image_height)
                for image_id in range(5):
                    image_list.append(img)
                    self.validation_annotation.append(word_table.img2sentence[file_name + '#' + str(image_id)])
                if count >= 20:
                    break
            self.validation_data = np.asarray(image_list)
        print
        print("validation data shape is : ", self.validation_data.shape)

    def save(self):
        """ save the dataset to pickle """
        pickle.dump([self.training_data, self.validation_data, self.training_annotation,
                     self.validation_annotation, self.batch_size, self.is_training, self.image_width, self.image_height],
                    open(self.save_file, 'wb'), protocol=4)

    def load(self):
        """ load the dataset from pickle """
        self.training_data, self.validation_data, self.training_annotation, self.validation_annotation, self.batch_size, self.is_training, self.image_width, self.image_height = pickle.load(open(self.save_file, 'rb'))

def prepare_data(arg):
    word_table_file = arg.wordtable_save
    glove_file_path = arg.glove_file_path
    vector_dim = arg.vector_dim
    data_file_path = arg.data_file_path

    """"build the word table """
    word_table = wordtable.WordTable(vector_dim, word_table_file)
    if not os.path.exists(word_table_file):
        word_table.load_gloves(glove_file_path)
        word_table.build(data_file_path)
        word_table.save()
    else:
        word_table.load()

    """"build the dataset """
    dataset = DataSet(arg.batch_size, True, arg.image_width, arg.image_height, arg.dataset_save)
    if not os.path.exists(arg.dataset_save):
        dataset.setup(arg.data_file_path, arg.num2train, word_table)
        dataset.save()
    else:
        dataset.load()

    return word_table, dataset

def resize_image(img, width, height):
    img = cv2.resize(img, (width, height))
    return img
