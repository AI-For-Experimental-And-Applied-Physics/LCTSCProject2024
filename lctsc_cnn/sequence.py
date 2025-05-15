import numpy as np
from tensorflow.keras.utils import Sequence
import os

class lctsc_sequence(Sequence):
    def __init__(self, data_path, batch_size=1, shuffle=True):
        self.data_path = data_path
        self.files = [f for f in os.listdir(data_path) if f.endswith('.npz')]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.files))
        self.on_epoch_end()

    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        # Generate batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        #batch_images = self.images[batch_indices]
        #batch_labels = self.labels[batch_indices]

        batch_files = [self.files[i] for i in batch_indices]
        batch_images = []
        batch_labels = []
        for f in batch_files:
            data = np.load(os.path.join(self.data_path, f))
            batch_images.append(data['image'])
            # Assuming the labels are stored in the same file
            # Adjust this if your labels are stored differently
            dump_label = np.zeros(data['image'].shape)
            dump_label[(data['Lung_R'] == 1)|(data['Lung_L'] == 1)] = 1
            batch_labels.append(dump_label)

        # Expand dims for channels (MNIST is grayscale)
        batch_images = np.expand_dims(batch_images, -1)
        batch_labels = np.expand_dims(batch_labels, -1)
        return batch_images, batch_labels
    
    def get_voxel_sizes(self):
        voxel_sizes= np.zeros((self.__len__(),3))
        # Load the first file to get the voxel size
        for idx,f in enumerate(self.files):
            voxel_sizes[idx] = np.load(os.path.join(self.data_path, f))['pixel_dim']
        
        return voxel_sizes
    
    def get_voxel_size(self,idx):
        return np.load(os.path.join(self.data_path, self.files[idx]))['pixel_dim']

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)