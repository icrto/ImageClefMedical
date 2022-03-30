import csv
import numpy as np
import matplotlib.pyplot as plt

concept_dict = {}
train_concept_dict = {}
valid_concept_dict = {}

with open('concepts.csv', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
  first_line = True
  for row in reader:
    if first_line:
      first_line = False
    else:
      concept_dict[row[0]] = row[1]
      train_concept_dict[row[0]] = []
      valid_concept_dict[row[0]] = []

with open('concept_detection_train.csv', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t')
  first_line = True
  for row in reader:
    if first_line:
      first_line = False
    else:
      image_name = row[0]
      labels = row[1].split(';')
      for label in labels:
        train_concept_dict[label].append(image_name)

num_images_concept = []
for concept in train_concept_dict.keys():
  num_images_concept.append(len(train_concept_dict[concept]))
  
print('TRAINING')
print('Total number of concepts: ' + str(len(num_images_concept)))
print('Concepts that do not appear in images: ' + str(len(np.where(np.asarray(num_images_concept) == 0)[0])))
print('Mean of images for each concept: ' + str(np.mean(num_images_concept)))
print('Min of images for each concept: ' + str(np.min(num_images_concept)))
print('Max of images for each concept: ' + str(np.max(num_images_concept)))
print('Concepts that appear in less than 10 images: ' + str(len(np.where(np.asarray(num_images_concept) <= 10)[0])))
print('Concepts that appear in more than 1000 images: ' + str(len(np.where(np.asarray(num_images_concept) > 1000)[0])))

top_n = 31
indexes_top = np.asarray(num_images_concept).argsort()[-top_n:][::-1]

top_concepts = np.asarray(list(train_concept_dict.keys()))[indexes_top]
to_show_concepts = []

for c in top_concepts:
  to_show_concepts.append(concept_dict[c])

to_show_values = [len(v) for v in np.asarray(list(train_concept_dict.values()))[indexes_top]]

fig=plt.figure(figsize=(12,8), dpi= 100)
plt.bar(to_show_concepts, to_show_values)
plt.xticks(rotation=45, ha="right")
plt.show()

with open('concept_detection_valid.csv', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t')
  first_line = True
  for row in reader:
    if first_line:
      first_line = False
    else:
      image_name = row[0]
      labels = row[1].split(';')
      for label in labels:
        valid_concept_dict[label].append(image_name)

num_images_concept = []
for concept in valid_concept_dict.keys():
  num_images_concept.append(len(valid_concept_dict[concept]))

print('VALIDATION')
print('Total number of concepts: ' + str(len(num_images_concept)))
print('Concepts that do not appear in images: ' + str(len(np.where(np.asarray(num_images_concept) == 0)[0])))
print('Mean of images for each concept: ' + str(np.mean(num_images_concept)))
print('Min of images for each concept: ' + str(np.min(num_images_concept)))
print('Max of images for each concept: ' + str(np.max(num_images_concept)))
print('Concepts that appear in less than 10 images: ' + str(len(np.where(np.asarray(num_images_concept) <= 10)[0])))
print('Concepts that appear in more than 100 images: ' + str(len(np.where(np.asarray(num_images_concept) > 100)[0])))

top_n = 31
indexes_top = np.asarray(num_images_concept).argsort()[-top_n:][::-1]

top_concepts = np.asarray(list(valid_concept_dict.keys()))[indexes_top]
to_show_concepts = []

for c in top_concepts:
  to_show_concepts.append(concept_dict[c])

to_show_values = [len(v) for v in np.asarray(list(valid_concept_dict.values()))[indexes_top]]

fig=plt.figure(figsize=(12,8), dpi= 100)
plt.bar(to_show_concepts, to_show_values)
plt.xticks(rotation=45, ha="right")
plt.show()