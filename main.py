import tensorflow as tf
import numpy as np
from model import ImageCaptioner
import argparse
import utils.preprocessing as pre

''' Retrieve all necessary parameters to train/test the ImageCaptioner '''
def get_parameters():
    parser = argparse.ArgumentParser()

    ##########################################
    ### Location of training/testing files ###
    ##########################################
    parser.add_argument(
        '--train_img_dir',
        help='Path to training images',
        required=True
    )
    parser.add_argument(
        '--train_captions',
        help='Path to training captions file',
        required=True
    )
    parser.add_argument(
        '--test_img_dir',
        help='Path to test images',
        required=True
    )
    parser.add_argument(
        '--test_captions',
        help='Path to testing captions file',
        required=True
    )

    ##########################
    ### Saving checkpoints ###
    ##########################
    parser.add_argument(
        '--ckpt_dir',
        help='Saves checkpoints here',
        default='./checkpoints/'
    )
    parser.add_argument(
        '--ckpt_freq',
        help='Save model after this many iterations',
        default=1000,
        type=int
    )

    ############################
    ### General model params ###
    ############################
    parser.add_argument(
        '--solver',
        help='Optimizer to use: {adam, sgd}',
        default='adam'
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate',
        default=1e-3,
        type=float
    )
    parser.add_argument(
        '--batch_size',
        help='Specify how many images to use for each iteration',
        default=20,
        type=int
    )
    parser.add_argument(
        '--num_epochs',
        help='Specify how many epochs to run for',
        default=20,
        type=int
    )

    ########################
    ### CNN model params ###
    ########################
    parser.add_argument(
        '--cnn_model',
        help='Type of CNN to use: {custom, vgg16}',
        default='vgg16'
    )
    parser.add_argument(
        '--cnn_model_file',
        help='Path to pretrained model file',
        required=True
    )
    parser.add_argument(
        '--train_cnn',
        help='Train CNN jointly with the RNN if flag is set',
        action='store_true',
        default=False
    )

    ########################
    ### RNN model params ###
    ########################
    parser.add_argument(
        '--hidden_size',
        help='Number of LSTM units to use',
        default=750,
        type=int
    )
    parser.add_argument(
        '--dim_embed',
        help='Dimension for the word embedding',
        default=300,
        type=int
    )
    parser.add_argument(
        '--dim_decoder',
        help='Dimension of the vector used for word generation',
        default=1000,
        type=int
    )

    args = parser.parse_args()
    return args


def main(_):

    # Retrieve parameters
    config = get_parameters()

    # Preprocess all images
    test_data = pre.load_image('data/laska.png', (224,224))
    test_data = np.array([test_data[:,:,:3]])
    
    # Build model.
    model = ImageCaptioner(config)

    # model.train(train_data)
    model.test(test_data)

if __name__ == "__main__":
    tf.app.run()
