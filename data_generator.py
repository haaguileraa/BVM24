import os
import numpy as np
import tensorflow as tf
import imageio.v2 as io
from PIL import Image
import albumentations as A


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, size=None, shuffle=False, batch_size = 32, augment = None):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.size = size
        self.shuffle = shuffle
        self.validation_indices = None
        self.training_indices = None
        self.augment = augment
        
        self.mask_paths =  [os.path.splitext(path)[0] + "_seg.png" for path in self.image_paths] ## for BAGLS
        #self.mask_paths = [path.replace("images", "masks") for path in self.image_paths] ## for kvasir-seg
        self._on_epoch_end()

    def __len__(self):
        return len(self.image_paths)

    def _generate_data(self, indices):
        images = np.zeros((self.batch_size, *self.size, 1))
        masks = np.zeros((self.batch_size, *self.size, 1))
        for i, bn in enumerate(indices):
            img_path = self.image_paths[bn]
            mask_path = img_path.replace('.png', '_seg.png')
            mask = io.imread(mask_path).astype('float32') 
            #if np.max(mask) != 0:
            img = Image.open(img_path).convert("L")  # Open image in grayscale
            if self.size is not None:        
                img = np.array(img.resize(self.size))#.astype('float32') / 255.0
                mask = np.array(Image.fromarray(mask).resize(self.size))#.astype('float32') / 255.0
            if self.augment is not None:
                augmented = self.augment(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            images[i,] = np.expand_dims(img, -1)
            masks[i,] = np.expand_dims(mask, -1)
        
        return images.astype('float32') / 255.0, masks.astype('float32') / 255.0


    def _on_epoch_end(self):
        
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            #np.random.seed(42)
            np.random.shuffle(self.indices)
        
    def __getitem__(self, bn):
        batch_ix = self.indices[bn * self.batch_size:(bn+1)*self.batch_size]
        return self._generate_data(batch_ix)
 

 