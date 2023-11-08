import time
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Activation, MaxPool2D, UpSampling2D, GroupNormalization, Add, Multiply
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
import albumentations as A
import json
TRAINING_PATH = "./training224x224/"
image_width = 224
image_height = 224


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


def unet(filters=8, layers=4, num_classes = 1, input_shape=(224,224,1), activation='swish'):
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
    
    if num_classes == 1:
        x = Conv2D(1, (1,1), padding='same')(x)
        model_out = Activation("sigmoid")(x)
        
        return Model(model_in, model_out)
    if num_classes == 2:
        x1 = Conv2D(1, (1,1), padding='same')(x)
        x2 = Conv2D(1, (1,1), padding='same')(x)
        x1 = Activation("sigmoid")(x1)
        x2 = Activation("sigmoid")(x2)
        return Model(model_in, (x1, x2))


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
    num_nests = [1, 2, 3, 4]
    numFilters = [8, 16] 
    operations = ["add", "multiply", "concatenate"]
    N = int(55749*0.1) # 10% of the dataset
    batch_size = 64
    validation_split = 0.2
    NUM_CLASSES = 1
    
    transforms = A.Compose([A.RandomBrightnessContrast(p=0.5),
                            A.HorizontalFlip(0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.HueSaturationValue(p=0.5),
                            ])


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
                
                #image_paths = [TRAINING_PATH + str(i) + ".png" for i in range(N)] ## for BAGLS
                image_paths = ["./kvasir-seg/images/" + f for f in os.listdir("./kvasir-seg/images")] ## for kvasir-seg
            
                train_paths, val_paths = train_test_split(image_paths, test_size=validation_split)
                
                train_data = DataGenerator(train_paths, (image_width, image_height), shuffle=True, batch_size=batch_size, augment=transforms)
                val_data = DataGenerator(val_paths, (image_width, image_height), shuffle=False, batch_size=batch_size)
                
                log_dir = f'./logs/{current_num_nests}_{current_num_filters}_{current_operation}_kvasir'
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                nested_model = nested_unet(nests=current_num_nests, filters=current_num_filters, operation=current_operation, input_shape=(image_width, image_height, 1))
                nested_model.compile(optimizer = Adam(learning_rate=1e-3),
                                     #loss= "mse", 
                                     loss = dice_loss,
                                     metrics=[iou_score])
                                                    
                model_checkpoint = ModelCheckpoint(
                    filepath=f"./checkpoints/nestedUnet_{current_num_nests}_{current_num_filters}_{{epoch}}_kvasir.h5",
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='min',
                    save_freq='epoch'
                )
                
                time_callback = TimeHistory()
                
                # Training
                history = nested_model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[model_checkpoint, stop_criterion, time_callback, tensorboard_callback])
                history_dict = history.history

                with open(f"./history/history_{current_num_nests}_{current_num_filters}_{current_operation}_kvasir.json", 'w') as f:
                    json.dump(history_dict, f)
                # # Save the final model using the native Keras format
                nested_model.save(f"./models/nestedUnet_{current_num_nests}_{current_num_filters}_{current_operation}_{1}_final_kvasir.keras")
                
                # Save epoch-wise time taken during training
                with open(f"./time_history/time_history_{current_num_nests}_{current_num_filters}_{current_operation}_kvasir.txt", "w") as f:
                    f.write("\n".join(str(t) for t in time_callback.times))
                
if __name__ == '__main__':
    main()