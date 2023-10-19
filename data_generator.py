import os
import numpy as np
import tensorflow as tf
import imageio.v2 as io
from PIL import Image

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path, batch_size, num_ims, size=None, is_training=False, validation_split=0.2):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.num_ims = num_ims
        self.size = size
        self.is_training = is_training
        self.validation_split = validation_split

        self.image_paths = [folder_path + str(i) + ".png" for i in range(num_ims)]
        if is_training:
            self.mask_paths = [folder_path + str(i) + "_seg.png" for i in range(num_ims)]
            self.masks = []
            for i in range(num_ims):
                mask = io.imread(self.mask_paths[i])
                if np.max(mask) != 0:
                    if np.max(mask) > 10:
                        mask = mask / 255.0
                    img = Image.open(self.image_paths[i]).convert("L")
                    if size is not None:
                        img = np.array(img.resize(size))
                        mask = np.array(Image.fromarray(mask).resize(size))
                    self.masks.append(np.array(tf.expand_dims(mask, -1)))

            self.output = np.array(self.masks)
            self.indices = np.arange(len(self.output))
            np.random.seed(42)
            np.random.shuffle(self.indices)
            split_index = int(len(self.output) * (1 - validation_split))
            self.train_indices = self.indices[:split_index]
            self.val_indices = self.indices[split_index:]
        else:
            self.output = []
            for i in range(len(self.image_paths)):
                img = Image.open(self.image_paths[i]).convert("L")
                if size is not None:
                    img = np.array(img.resize(size))
                self.output.append(np.array(tf.expand_dims(img, -1)))
            self.output = np.array(self.output)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("L")
        if self.size is not None:
            img = img.resize(self.size)
        img = np.array(img)
        img = np.expand_dims(img, -1)
        img = img / 255.0
        return img

    def get_dataset(self, indices):
        images = [self.preprocess_image(image_path) for image_path in np.array(self.image_paths)[indices]]
        images = np.array(images)
        labels = self.output[indices]
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(len(images))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_train_dataset(self):
        return self.get_dataset(self.train_indices)

    def get_val_dataset(self):
        return self.get_dataset(self.val_indices)

    def __len__(self):
        return int(np.ceil(len(self.output) / float(self.batch_size)))

    def __getitem__(self, idx):
        dataset = self.get_train_dataset()
        batch = dataset.__iter__().get_next()
        return batch