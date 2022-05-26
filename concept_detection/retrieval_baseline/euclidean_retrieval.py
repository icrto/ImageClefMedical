import os, math, pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow"
import h5py, numpy as np, time, shutil, random
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, BatchNormalization, LeakyReLU, Lambda, MaxPooling2D, Dot, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow.keras.backend as K
from load_data_sequence import ImageClefDataset
import sklearn.metrics

num_labels = 32
embed_size = 128
epochs = 1000
opt = Adam(lr=1e-6)
cosine_sim = True
train_shape = (224, 224, 1)

concepts_filename = 'dataset_resized/new_top32_concepts.csv'
concepts_csv = pd.read_csv(concepts_filename, sep="\t")
all_concepts = concepts_csv["concept"]
concepts = []
dict_concept = dict()
dict_concept_inverse = dict()
for idx, c in enumerate(all_concepts):
    dict_concept[c] = idx
    dict_concept_inverse[idx] = c
    concepts.append(to_categorical(idx, len(all_concepts)))
        
concepts = np.asarray(concepts)
print(concepts.shape)

train_data = ImageClefDataset(
    dict_concept,
    'dataset_resized/new_train_subset_top32.csv',
    'dataset_resized/train_resized', 0,
    concepts, num_labels
)

valid_data = ImageClefDataset(
    dict_concept,
    'dataset_resized/new_train_subset_top32.csv',
    'dataset_resized/train_resized', 1,
    concepts, num_labels
)

test_data = ImageClefDataset(
    dict_concept,
    'dataset_resized/new_val_subset_top32.csv',
    'dataset_resized/valid_resized', 2,
    concepts, 1
)

infer_data = ImageClefDataset(
    dict_concept,
    'dataset_resized/concept_detection_valid.csv',
    'dataset_resized/valid_resized', 3,
    concepts, 1
)

# Build models

def build_feature_extractor():
    img_input = Input(train_shape)
    h1 = Conv2D(int(embed_size / 8), (5, 5), activation = 'relu', padding = 'same', name = 'id_conv1')(img_input)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(int(embed_size / 4), (5, 5), activation = 'relu', padding = 'same', name = 'id_conv2')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(int(embed_size / 2), (3, 3), activation = 'relu', padding = 'same', name = 'id_conv3')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    h1 = Conv2D(embed_size, (3, 3), activation = 'relu', padding = 'same', name = 'id_conv4')(h1)
    h1 = BatchNormalization()(h1)
    h1 = MaxPooling2D((3, 3), padding='same', strides = (2, 2))(h1)

    features = GlobalAveragePooling2D()(h1)
    features = Dense(embed_size, activation = 'tanh', name = 'medical_features')(features)

    feat_extractor = Model(img_input, features)

    return feat_extractor

def build_label_encoder():
    label_input = Input((len(all_concepts),))
    x = Dense(embed_size/2)(label_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Dense(embed_size, activation = 'tanh')(x)
    label_encoder = Model(label_input, x)
    return label_encoder

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
      l.trainable = val

def contrastive_loss(y_true, y_pred, margin=1):
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def distance_lambda(image_encodings, label_encodings):
    A_dots = K.sum(image_encodings*image_encodings, axis=1, keepdims=True)*np.ones(shape=(1,num_labels))
    B_dots = K.sum(label_encodings*label_encodings, axis=1)*np.ones(shape=(num_labels,1))
    D_squared = A_dots + B_dots -2*K.dot(image_encodings, K.transpose(label_encodings))
    return K.sqrt(D_squared)

input_imgs = Input(train_shape)
feature_extractor = build_feature_extractor()
label_encoder = build_label_encoder()

input_img = Input(train_shape)
input_label = Input((len(all_concepts),))
feat_img = feature_extractor(input_img)
feat_label = label_encoder(input_label)

distance = Lambda(lambda x: distance_lambda(x[0], x[1]), output_shape=(num_labels,))([feat_img, feat_label])

training_model = Model([input_img, input_label], [distance])
training_model.compile(optimizer=opt, loss=contrastive_loss)


path = "./models"
if os.path.isdir(path) == True:
  shutil.rmtree(path)
os.mkdir(path)

checkpoint_1 = ModelCheckpoint('./models/eucl_training_weights_{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
checkpoint_2 = ModelCheckpoint('./models/e_training_weights_{epoch:03d}.h5', period=50) 

training_model.fit_generator(train_data, validation_data=valid_data, callbacks=[checkpoint_1,checkpoint_2], epochs=epochs, steps_per_epoch=len(train_data), workers=1)
#training_model.save_weights('./models/e_training_weights_final.h5')


labels = range(0, len(all_concepts))
labels = to_categorical(labels, len(all_concepts))
encoded_labels = label_encoder.predict(labels)

def np_euclidean_distance(image_encodings, label_encodings):
    dist = []
    for lab in label_encodings:
      dist.append(np.sqrt(np.sum(np.square(image_encodings[0] - lab))))
    return np.asarray(dist)

def evaluate():
    contains = 0
    euclidean_distance_matrix = []
    y_true = []
    top5_predicted_labels = []
    top3_predicted_labels = []
    top1_predicted_labels = []
    for idx in range(0, len(test_data)):    
        test_data_img, test_data_lbl = test_data[idx]
        y_true.append(test_data_lbl)
        image_features = feature_extractor.predict(test_data_img)
        distance_image = np_euclidean_distance(image_features, encoded_labels)
        euclidean_distance_matrix.append(distance_image)
        top5 = distance_image.argsort()[:5]
        top3 = top5[:3]
        top1 = top5[0]
        #print(top1)
        pred = np.zeros((len(all_concepts),))
        pred[top5] = 1
        top5_predicted_labels.append(pred)
        pred = np.zeros((len(all_concepts),))
        pred[top3] = 1
        top3_predicted_labels.append(pred)
        pred = np.zeros((len(all_concepts),))
        pred[top1] = 1
        top1_predicted_labels.append(pred)
        
        if test_data_lbl[0][top1] == 1:
            contains += 1
    
    norm_matrix = (euclidean_distance_matrix-np.min(euclidean_distance_matrix))/np.max(euclidean_distance_matrix-np.min(euclidean_distance_matrix))

    min_array = []
    for i in range(len(norm_matrix)):
        min_array.append(np.min(norm_matrix[i]))
        
    avg = np.average(min_array)
    std = np.std(min_array)
    threshold = avg + std*0#.2
    print(threshold)
    y_pred = np.where(norm_matrix < threshold, 1, 0)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
    for i in range(len(y_pred)):
        y_pred[i][np.where(top1_predicted_labels[i] == 1)[0]] = 1
    y_true = np.asarray(y_true)
    y_true = np.reshape(y_true, (y_true.shape[0], y_true.shape[-1]))
    print(y_pred.shape)
    print('Exact Match Ratio: ' + str(sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))
    print('Hamming loss: ' + str(sklearn.metrics.hamming_loss(y_true, y_pred)))
    print('Recall: ' + str(sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('Precision: ' + str(sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples')))
    print('F1-Score Top 5: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=np.asarray(top5_predicted_labels), average='samples')))
    print('F1-Score Top 3: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=np.asarray(top3_predicted_labels), average='samples')))
    print('F1-Score Top 1: ' + str(sklearn.metrics.f1_score(y_true=y_true, y_pred=np.asarray(top1_predicted_labels), average='samples')))
    print('Percentage of closest concepts that exist in the labels: ' + str(contains/len(test_data)))

def infer():
    euclidean_distance_matrix = []
    top1_predicted_labels = []
    eval_concepts = []
    eval_images = []
    for idx in range(0, len(infer_data)):
        eval_images.append(infer_data.image_names[idx])
        infer_data_img, infer_data_lbl = infer_data[idx]
        image_features = feature_extractor.predict(infer_data_img)
        distance_image = np_euclidean_distance(image_features, encoded_labels)
        euclidean_distance_matrix.append(distance_image)
        top1 = distance_image.argsort()[0]
        pred = np.zeros((len(all_concepts),))
        pred[top1] = 1
        top1_predicted_labels.append(pred)

    norm_matrix = (euclidean_distance_matrix-np.min(euclidean_distance_matrix))/np.max(euclidean_distance_matrix-np.min(euclidean_distance_matrix))

    y_pred = np.where(norm_matrix < threshold, 1, 0)
    y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
    for i in range(len(y_pred)):
        img_concepts = []
        y_pred[i][np.where(top1_predicted_labels[i] == 1)[0]] = 1
        all_ones = np.where(y_pred[i] == 1)[0]
        for idx in all_ones:
            img_concepts.append(dict_concept_inverse[idx])
        eval_concepts.append(';'.join(img_concepts))
    eval_set = dict()
    eval_set["ID"] = eval_images
    eval_set["cuis"] = eval_concepts

    evaluation_df = pd.DataFrame(data=eval_set)
    # VALID
    evaluation_df.to_csv('./infer_results.csv', sep='\t', index=False)
    # TEST
    #evaluation_df.to_csv('./infer_results.csv', sep='|', index=False, header=False)

threshold = 0
evaluate()
infer()