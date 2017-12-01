from scipy.misc import imread, imresize

def load_image(path, shape):
    img = imread(path, mode='RGB')
    img = imresize(img, shape)
    return img
