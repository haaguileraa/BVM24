import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio.v2 as io
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, MaxPool2D, UpSampling2D, GroupNormalization, \
                                    Add, Multiply
from tensorflow.keras.models import Model

TRAINING_PATH = "./training224x224/"
image_width = 224
image_height = 224
NUM_NESTS = 4
NUM_FILTERS = 8


def conv(x, filters:int=8, activation:str="swish"):
    for i in range(2):
        x = Conv2D(filters, (3,3), padding='same')(x)
        x = GroupNormalization(groups=-1)(x) # InstanceNorm
        x = Activation(activation)(x)
    return x


def unet(filters=8, layers=4, input_shape=(224,224,1), activation='swish'):
    to_concat = []
    
    model_in = Input(input_shape)
    x = model_in
    
    # Encoder
    for i in range(layers):
        x = conv(x, filters*2**i, activation)
        to_concat.append(x)
        x = MaxPool2D()(x)
    
    # Latent
    x = Conv2D(filters*2**layers, (3,3), padding='same')(x)
    x = GroupNormalization(groups=-1)(x)  # InstanceNorm
    x = Activation(activation)(x)
    
    # Decoder
    for i in range(layers):
        x = UpSampling2D()(x)
        x = Concatenate()([x, to_concat.pop()])
        x = conv(x, filters*2**(layers-i-1), activation)
    
    x = Conv2D(1, (1,1), padding='same')(x)
    model_out = Activation("sigmoid")(x)
    
    return Model(model_in, model_out)


def nestedUnet(nests=4, filters=1, forward_input=True, operation="multiply", input_shape=(256, 256, 1)):
    x = Input(input_shape)
    m0 = unet(filters, input_shape=input_shape)(x)
    
    if nests > 1:
        tmp = m0
        
        for i in range(nests-1):
            if forward_input:
                if operation == 'add':
                    tmp = Add()([x, tmp])
                    
                elif operation == 'multiply':
                    tmp = Multiply()([x, tmp])
                    
                else:
                    tmp = Concatenate(axis=3)([x, tmp])
                
            tmp = unet(filters, input_shape=tmp.shape[1:])(tmp)

        return Model(x, tmp)        
        
    else:
        return Model(x, m0)

if __name__ == '__main__':
    
    N = 22 # arbitrary number of samples

    train_imgs = [TRAINING_PATH + str(i) + ".png" for i in range(N)]
    train_segs = [TRAINING_PATH + str(i) + "_seg.png" for i in range(N)]

    imgs = []
    segs = []

    for i in range(N):
        seg = io.imread(train_segs[i])
        if np.max(seg) != 0:  # To remove the black images
            if np.max(seg) > 10:
                seg = seg / 255.0

            img = io.imread(train_imgs[i]) / 255.0

            imgs.append(np.expand_dims(img, axis=2))
            segs.append(np.expand_dims(seg, axis=2))

            # imgs.append(np.array(tf.expand_dims(img, -1)))
            # segs.append(np.array(tf.expand_dims(seg, -1)))

    # Randomizing
    nSamples = len(imgs)
    np.random.seed(42)
    indices = np.arange(nSamples)
    np.random.shuffle(indices)  # Shuffling indices directly

    imgs = [imgs[i] for i in indices]
    segs = [segs[i] for i in indices]


    nestedModel = nestedUnet(nests=NUM_NESTS, filters=NUM_FILTERS, input_shape=(image_width, image_height, 1))
    nestedModel.summary()
    nestedModel.compile("adam", "mse")

    # Training
    history = nestedModel.fit(np.array(imgs), np.array(segs), epochs=10, batch_size=4)
