"""
Loads images, makes targets, runs them through
"""
import os
import itertools
import numpy as np

import keras
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions


mobilenet = MobileNet(weights='imagenet', include_top=False)

IMG_DIR = '/Users/larshertel/Downloads/PolyvoreImages/new/mix/'
DATA_DIR = '/Users/larshertel/Downloads/PolyvoreImages/embeddings/'

# paths = os.listdir(path)[1:]

def build_model():
    input_x = Input(shape=(None, 1024)) # N x L x 1024
    input_z = Input(shape=(1024,))  # N x 1024

    z = Dense(128, activation='tanh')(input_z) # N x 128
    z = Dense(128, activation='tanh')(z)

    def avg(x):
        return K.mean(input_x, axis=1, keepdims=False)

    # x = Lambda(avg, output_shape=(1024,))(input_x) # N x 128
    # x = Dense(128, activation='tanh')(x) # N x 128

    x = LSTM(32)(input_x)
    x = Dense(128, activation='tanh')(x) # N x 128

    def outer_prod(x):
        u = K.l2_normalize(x[0], axis=1)
        v = K.l2_normalize(x[1], axis=1)
        return K.dot(u, K.transpose(v))

    out = Lambda(outer_prod, output_shape=(None,))([x, z]) # N x N

    yhat = Activation('softmax')(out)

    model = Model([input_x, input_z], yhat)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def embed_outfits(paths, postprocess=lambda x: x):
    """
    postprocess = lambda x: x.mean(axis=0).mean(axis=0).squeeze()
    """
    mobilenet = MobileNet(weights='imagenet', include_top=False)
    outfit_names, outfit_imgs = [], []
    g = itertools.groupby(sorted(paths), key=lambda item: item[:-9])
    for outfit_name, img_seq in g:
        print(outfit_name)
        img_arrays = []
        for img_name in img_seq:
            img_path = IMG_DIR + img_name
            img = image.load_img(img_path, target_size=(224, 224))
            img_arrays.append(image.img_to_array(img))
        if len(img_arrays) == 5:
            x = np.array(img_arrays)
            x = preprocess_input(x)
            features = mobilenet.predict(x)
            features = postprocess(features)
            outfit_imgs.append(features)
            outfit_names.append(outfit_name)
    return np.array(outfit_names), np.array(outfit_imgs)


# store embeddings as Nx5xP array
def save_outfit_data(outfit_names, outfit_imgs, path):
    names_path = os.path.join(path, 'outfit_names')
    imgs_path = os.path.join(path, 'outfit_imgs')
    np.save(names_path, outfit_names)
    np.save(imgs_path, outfit_imgs)

def load_outfit_data(path):
    names = np.load(os.path.join(path, 'outfit_names.npy'))
    imgs = np.load(os.path.join(path, 'outfit_imgs.npy'))
    return names, imgs

def batch_generator(img_features, batch_size=32):
    N, L, feature_dim = img_features.shape
    while True:
        l = np.random.randint(2, L)
        idxs = np.random.choice(list(range(N)),
                                batch_size,
                                replace=False)

        # want idxs along first axis and then l samples in [2,3,4,5]
        # for second axis
        axis_0_idxs = np.repeat(idxs, l)
        # list of idxs repeated by l: np.repeat(idxs, l)
        # idx1item1, idxs1item2, idxs2item1,...
        axis_1_idxs = np.concatenate([np.random.choice(list(range(5)),
                                                       l, replace=False)
                                                       for _ in range(batch_size)])
        arr = img_features[axis_0_idxs, axis_1_idxs, :].reshape(batch_size, l, feature_dim)
        yield [arr[:, :-1, :], arr[:, -1, :]], np.eye(batch_size)

def store_data():
    paths = os.listdir(IMG_DIR)
    postprocess = lambda x: x.mean(axis=1).mean(axis=1)
    names, imgs = embed_outfits(paths,
                                postprocess=postprocess)
    print("TOTAL DATA SHAPE: {}".format(imgs.shape))
    save_outfit_data(names, imgs, DATA_DIR)



if __name__ == '__main__':
    if not os.path.isfile(os.path.join(DATA_DIR, 'outfit_imgs.npy')):
        store_data()
    names, imgs = load_outfit_data(DATA_DIR)
    model = build_model()
    num_train = int(imgs.shape[0]*4/5)
    g = batch_generator(imgs[:num_train], batch_size=32)
    v = batch_generator(imgs[num_train:], batch_size=32)
    model.fit_generator(g, steps_per_epoch=5000,
                        epochs=10, validation_data=v,
                        validation_steps=500)
