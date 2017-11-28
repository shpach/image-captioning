from skimage import io, transform

def load_image(path, shape):
    img = io.imread(path, mode='RGB')
    img = transform.resize(img, shape)
    return img
