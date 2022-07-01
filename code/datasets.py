from random import shuffle
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

dataset_split = {
    "oxford_iiit_pet" : ['train[:90%]', 'train[90:100%]', 'test'],
    "oxford_flowers102" : ['test', 'train', 'validation'],
    "tf_flowers" : ['train[:80%]', 'train[80%:90%]', 'train[90%:100%]'],
    "uc_merced" : ['train[:70%]', 'train[70%:85%]', 'train[85%:100%]'],
    "rock_paper_scissors" : ['train[:90%]', 'train[90:100%]', 'test'],
    "cars196" : ['train', 'test[:15%]', 'test[15%:]'],
    "caltech101" : ['test[:90%]', 'test[90%:]', 'train'],
    "food101" : ['train[:10100]', 'train[10100:11200]', 'train[11200:21300]']
}

def get_supported_datasets():
    return tuple(dataset_split.keys())

def load_data(dataset_name):
    if dataset_name not in dataset_split:
        raise ValueError("unknown dataset name")
    
    (data_train, data_val, data_test), info = tfds.load(dataset_name,
                                                    split=dataset_split[dataset_name],
                                                    with_info=True,
                                                    as_supervised=True,
                                                    shuffle_files=True)

    return data_train, data_val, data_test, info


def get_resize_image(img_size=224):
    def resize(image):
        return tf.image.resize(image, (img_size, img_size))
    return resize

augment_layer = keras.models.Sequential([
    keras.layers.RandomFlip(),
    keras.layers.RandomRotation(factor=0.2),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    
    keras.layers.RandomContrast(factor=0.1)
])

def preprocess_data(data_train, data_val, data_test, img_size=224, batch_size=32, prefetch=True):
    resize = get_resize_image(img_size)
    
    data_train = data_train.shuffle(500)
    data_train = data_train.map(lambda img, lab : (resize(img), lab))
    data_train = data_train.map(lambda img, lab : (augment_layer(img), lab))
    data_train = data_train.batch(batch_size)
    
    if prefetch:
        data_train = data_train.prefetch(1)
    
    data_val = data_val.map(lambda img, lab : (resize(img), lab)).batch(batch_size)
    if prefetch:
        data_val = data_val.prefetch(1)


    data_test = data_test.map(lambda img, lab : (resize(img), lab)).batch(batch_size)
    if prefetch:
        data_test = data_test.prefetch(1)

    return data_train, data_val, data_test