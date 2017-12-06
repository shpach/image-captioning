from scipy.misc import imread, imresize

def load_image(path, shape):
    img = imread(path, mode='RGB')
    img = imresize(img, shape)
    return img

def resize_images(data, shape):
	for train_idx in range(len(data.training_data)):
		train_img = data.training_data[train_idx]
		data.training_data[train_idx] = imresize(train_img, shape)

	for test_idx in range(len(data.validation_data)):
		test_img = data.validation_data[test_idx]
		data.validation_data[test_idx] = imresize(test_img, shape)
