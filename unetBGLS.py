import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio.v2 as io
from PIL import Image

from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, MaxPool2D, UpSampling2D, GroupNormalization, \
                                    Add, Multiply
from tensorflow.keras.models import Model

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

def nested_unet(nests=4, filters=1, forward_input=True, operation="multiply", input_shape=(256, 256, 1)):
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
    

def read_dataset(folder_path: str, num_ims: int, size: tuple = None, is_training: bool = False, randomize: bool = True, validation_split: float = 0.2) -> tuple:
    output = []
    image_paths = [folder_path + str(i) + ".png" for i in range(num_ims)]
    
    if is_training:
        mask_paths = [folder_path + str(i) + "_seg.png" for i in range(num_ims)]
        masks = []

        for i in range(num_ims):
            mask = io.imread(mask_paths[i])

            if np.max(mask) != 0:
                if np.max(mask) > 10:
                    mask = mask / 255.0

                img = Image.open(image_paths[i]).convert("L")  # Open image in grayscale

                if size is not None:
                    img = np.array(img.resize(size))
                    mask = np.array(Image.fromarray(mask).resize(size))

                output.append(np.array(tf.expand_dims(img, -1)))
                masks.append(np.array(tf.expand_dims(mask, -1)))

        if randomize:
            # Randomizing
            n_samples = len(output)
            np.random.seed(42)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            output = [output[i] for i in indices]
            masks = [masks[i] for i in indices]

        # Split into training and validation sets
        split_index = int(len(output) * (1 - validation_split))
        train_output, val_output = output[:split_index], output[split_index:]
        train_masks, val_masks = masks[:split_index], masks[split_index:]

        return (train_output, train_masks), (val_output, val_masks)
    else:
        for i in range(len(image_paths)):
            img = Image.open(image_paths[i]).convert("L")  # Open image in grayscale

            if size is not None:
                img = np.array(img.resize(size))

            output.append(np.array(tf.expand_dims(img, -1)))

        return output
    
def main():
    TRAINING_PATH = "./training224x224/"
    image_width = 224
    image_height = 224
    num_nests = [1, 2, 4, 8, 16]
    numFilters = [8, 16] 
    operations = ["add", "multiply", "concatenate"]
    N = 22 # arbitrary number of samples

    train_data, val_data = read_dataset(folder_path = TRAINING_PATH, num_ims = N, is_training = True, randomize = True)
    train_imgs, train_segs = train_data
    val_imgs, val_segs = val_data

    for current_num_nests in num_nests:
        for current_num_filters in numFilters:
            for current_operation in operations:
                
                nested_model = nested_unet(nests=current_num_nests, filters=current_num_filters, operation=current_operation, input_shape=(image_width, image_height, 1))
                nested_model.compile("adam", "mse")

                # Training
                
                history = nested_model.fit(np.array(train_data[0]), np.array(train_data[1]), epochs=10, batch_size=4)

                # Checkpoints
                nested_model.save(f"nestedUnet_{current_num_nests}_{current_num_filters}_{current_operation}.h5")