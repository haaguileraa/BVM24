import os
import numpy as np
import tensorflow as tf
import imageio.v2 as io
from PIL import Image

import tensorflow as tf
from PIL import Image
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, num_samples, image_size, is_training=True, validation_split=0.2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.is_training = is_training
        self.validation_split = validation_split
        
        self.image_paths = self.load_image_paths()
        self.train_indices, self.val_indices = self.split_train_val_indices()
        
    def load_image_paths(self):
        image_paths = []
        with open(self.data_path, "r") as f:
            for line in f:
                image_path = line.strip()
                image_paths.append(image_path)
        return image_paths[:self.num_samples]
    
    def split_train_val_indices(self):
        num_train_samples = int(len(self.image_paths) * (1 - self.validation_split))
        train_indices = np.arange(num_train_samples)
        val_indices = np.arange(num_train_samples, len(self.image_paths))
        return train_indices, val_indices
    
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("L")
        if self.image_size is not None:
            img = img.resize(self.image_size)
        img = np.array(img)
        img = np.expand_dims(img, -1)
        img = img / 255.0
        return img
    
    def get_dataset(self, indices):
        image_paths = [self.image_paths[i] for i in indices]
        images = [self.preprocess_image(image_path) for image_path in image_paths]
        images = np.array(images)
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.shuffle(len(images))
        dataset = dataset.batch(self.batch_size)
        return dataset
    
    def get_train_dataset(self):
        return self.get_dataset(self.train_indices)
    
    def get_val_dataset(self):
        return self.get_dataset(self.val_indices)
    
    def __len__(self):
        if self.is_training:
            return int(np.ceil(len(self.train_indices) / float(self.batch_size)))
        else:
            return int(np.ceil(len(self.val_indices) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        if self.is_training:
            dataset = self.get_train_dataset()
        else:
            dataset = self.get_val_dataset()
        batch = dataset.__iter__().get_next()
        return batch, batch