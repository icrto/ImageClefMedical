from tqdm import tqdm
import tensorflow as tf, numpy as np, math, pandas as pd, matplotlib
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img, Iterator

num_concepts = 32

class ImageClefDataset(Sequence):
    # dset: train (0), valid (1) or test (2)
    def __init__(self, dict_concept, data_filename, data_folder, dset, concepts, batch_size):
        data_csv = pd.read_csv(data_filename, header=0, sep="\t")
        all_image_names = data_csv[:]['ID']
        all_labels = np.array(data_csv.drop(['ID'], axis=1))
        self.concepts = concepts
        self.batch_size = batch_size
        self.dset = dset

        self.data_folder = data_folder
        self.label_matrix = np.zeros((len(all_image_names), len(self.concepts)))
        if dset != 3:
            for i in range(len(all_image_names)):
                concepts_per_image = data_csv["cuis"][i].split(";")
                for c in concepts_per_image:
                    self.label_matrix[i][dict_concept[c]] = 1

        print(self.label_matrix.shape)
        train_ratio = int(0.85 * len(data_csv))
        valid_ratio = len(data_csv) - train_ratio
        self.image_names = list(all_image_names)
        self.labels = list(all_labels)
        self.shuffle = False

        # TRAIN
        if dset == 0:
            print("Number of training images: " + str(train_ratio))
            self.image_names = self.image_names[:train_ratio]
            self.labels = self.labels[:train_ratio]
            self.shuffle = True

        # VALIDATION
        elif dset == 1:
            print("Number of validation images: " + str(valid_ratio))
            self.image_names = self.image_names[-valid_ratio:]
            self.labels = self.labels[-valid_ratio:]
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        data = []
        targets = []
        images = []

        chosen_concepts = range(len(self.concepts))
        if self.dset < 2:
            chosen_concepts = np.random.choice(range(len(self.concepts)), num_concepts, replace=False)
        
        for i in indexes:
            image = load_img(self.data_folder + '/' + self.image_names[i] + '.jpg',
                grayscale=True,
                target_size=(224, 224),
            )
            image = img_to_array(image)
            image = (image - 127.5) / 127.5
            image = np.reshape(image, (224, 224, 1))
            images.append(image)
            targets.append(self.label_matrix[i][chosen_concepts])

        if self.dset >= 2:
            return np.asarray(images), np.asarray(targets)

        return [np.asarray(images), self.concepts[chosen_concepts]], np.asarray(targets)
