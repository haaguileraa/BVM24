import os
import numpy as np
import tensorflow as tf
import imageio.v2 as io
from PIL import Image

class DataGenerator:
    def __init__(self, folder_path, num_samples, batch_size, image_size, shuffle=True, validation_split=0.2):
        self.folder_path = folder_path
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.indexes = np.arange(num_samples)
        self.train_indexes, self.val_indexes = self._split_train_val()
        self.train_steps = int(np.ceil(len(self.train_indexes) / self.batch_size))
        self.val_steps = int(np.ceil(len(self.val_indexes) / self.batch_size))
        
    def _split_train_val(self):
        split_index = int(self.num_samples * (1 - self.validation_split))
        train_indexes = self.indexes[:split_index]
        val_indexes = self.indexes[split_index:]
        print(f"Training on {len(train_indexes)} samples, validating on {len(val_indexes)} samples")
        return train_indexes, val_indexes
        
    def _load_image(self, path):
        img = Image.open(path).convert("L")
        if self.image_size is not None:
            img = img.resize(self.image_size)
        return np.array(tf.expand_dims(img, -1))
        
    def _load_mask(self, path):
        mask = io.imread(path)
        if np.max(mask) != 0:
            if np.max(mask) > 10:
                mask = mask / 255.0
            if self.image_size is not None:
                mask = np.array(Image.fromarray(mask).resize(self.image_size))
        return np.array(tf.expand_dims(mask, -1))
        
    def _get_data(self, indexes):
        X = []
        Y = []
        for i in indexes:
            image_path = os.path.join(self.folder_path, str(i) + ".png")
            mask_path = os.path.join(self.folder_path, str(i) + "_seg.png")
            X.append(self._load_image(image_path))
            Y.append(self._load_mask(mask_path))
            # print shapes
        print(f"X: {np.array(X).shape}, Y: {np.array(Y).shape}")
        return np.array(X), np.array(Y)
        
    def get_train_data(self):
        if self.shuffle:
            np.random.shuffle(self.train_indexes)
        while True:
            for i in range(self.train_steps):
                indexes = self.train_indexes[i*self.batch_size:(i+1)*self.batch_size]
                yield self._get_data(indexes)
            
    def get_validation_data(self):
        if self.shuffle:
            np.random.shuffle(self.val_indexes)
        while True:
            for i in range(self.val_steps):
                indexes = self.val_indexes[i*self.batch_size:(i+1)*self.batch_size]
                yield self._get_data(indexes)[0:2] # return only the first two values