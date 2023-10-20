import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, MaxPool2D, UpSampling2D, GroupNormalization, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import IoU

from data_generator import DataGenerator
from sklearn.model_selection import train_test_split

TRAINING_PATH = "./training224x224/"
image_width = 224
image_height = 224
NUM_NESTS = 4
NUM_FILTERS = 8


class TimeHistory(Callback):
    """Saving time epoch wise."""
    # https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


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


def main():
    image_width = 224
    image_height = 224
    num_nests = [1, 2, 4, 8, 16]
    numFilters = [8, 16] 
    operations = ["add", "multiply", "concatenate"]
    N = int(55749*0.1) # 10% of the dataset
    batch_size = 32
    validation_split = 0.2
    NUM_CLASSES = 1
    


    stop_criterion = EarlyStopping(
                                    monitor="val_loss",
                                    min_delta=0.001,
                                    verbose=1,
                                    patience=8,
                                    mode="min",
                                  )
    
    for current_num_nests in num_nests:
        for current_num_filters in numFilters:
            for current_operation in operations:
                
                image_paths = [TRAINING_PATH + str(i) + ".png" for i in range(N)]
                
                        
                train_paths, val_paths = train_test_split(image_paths, test_size=validation_split)
                

                train_data = DataGenerator(train_paths, N, (image_width, image_height), batch_size=batch_size)
                val_data = DataGenerator(val_paths, N, (image_width, image_height), batch_size=batch_size)
                
                

                nested_model = nested_unet(nests=current_num_nests, filters=current_num_filters, operation=current_operation, input_shape=(image_width, image_height, 1))
                nested_model.compile("adam", "mse",
                                      metrics=[IoU(num_classes=NUM_CLASSES, target_class_ids=[0])])
                
                model_checkpoint = ModelCheckpoint(
                    filepath=f"./checkpoints/nestedUnet_{current_num_nests}_{current_num_filters}_{{epoch}}.h5",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=False,
                    verbose=1
                )
                
                time_callback = TimeHistory()
                
                # Training
                history = nested_model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[model_checkpoint, stop_criterion, time_callback])

                # # Save the final model using the native Keras format
                nested_model.save(f"nestedUnet_{current_num_nests}_{current_num_filters}_{current_operation}_{1}_final.keras")
                
                # Save epoch-wise time taken during training
                with open(f"time_history_{current_num_nests}_{current_num_filters}_{current_operation}.txt", "w") as f:
                    f.write("\n".join(str(t) for t in time_callback.times))
if __name__ == '__main__':
    main()