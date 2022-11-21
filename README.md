# ImageClefMedical

This is the official repository for the [VCMI](https://vcmi.inesctec.pt)'s Team submission to [ImageClefmedical Caption 2022](https://www.imageclef.org/2022/medical/caption).

You can find the paper [here](http://ceur-ws.org/Vol-3180/paper-116.pdf).

For more information please contact isabel.riotorto@inesctec.pt.

## Requirements

You can find the package requirements in the requirements.txt file.

For pycocoevalcap make sure that your locale has en_US.UTF-8, otherwise computing METEOR will throw an error. To change your locale just do ```sudo update-locale LC_ALL=en_US.UTF-8``` and reboot your computer.

## Dataset Structure

```
ImageClefMedical/dataset
    train/
    valid/
    test/
    caption_prediction_train.csv
    caption_prediction_valid.csv
    concept_detection_train.csv
    concept_detection_valid.csv
    concepts.csv
```    
## Preprocessing

### Top-K most-frequent concepts

Run ```python preprocessing/get_topconcepts.py```. This script will create the following files:
1. concepts_top100.csv
2. concept_detection_train_top100.csv e concept_detection_valid_top100.csv
3. caption_prediction_train_top100.csv e caption_prediction_valid_top100.csv

The concepts_top100.csv corresponds to the concepts.csv file, but filtered to contain only the top-K most-frequent concepts. The other files correspond to their original counterparts but with the concepts not present in the top-K removed (images that end up without any valid concept are also removed).

### Conversion to COCO format

Run ```
python preprocessing/convert_to_coco.py <your_file_here>```. This script will convert the given file into COCO format and save it as <your_file_here>_coco.json.

You should call this script for both caption_prediction_train.csv and caption_prediction_valid.csv files (and/or their top-K versions).

### Generate csv for test images
Run ```python preprocessing/gen_test_images_csv.py --datadir dataset/test``` to generate a csv file with all the test images. This file is needed to generate predictions for all the test images.

## Concept Detection

### Multilabel

To train the multilabel model run ```python concept_detection/multilabel/train.py```. If you want to specify the number of top-K concepts to use, just add ```--nr_concepts <number_of_concepts_to_consider>```.

To make predictions, use the ```inference.py``` script. This script has two important arguments: the images_csv and nr_concepts. The first specifies for which images you want to generate predictions. For example, you might want to generate predictions only for the images that contain at least one of the top-100 concepts. The nr_concepts argument must be set in accordance to what the model was trained with. So, if for example, you trained your model to consider only the top-100 concepts, then nr_concepts should be 100. Then, for inference, two situations arise:
1. you want to generate predictions for the subset of images of the top-100 concepts: ```python inference.py --nr_concepts 100 --images_csv dataset/concept_detection_valid_top100.csv```
2. although your model was trained with 100 concepts you want to generate predictions for all validation images: ```python inference.py --nr_concepts 100 --images_csv dataset/concept_detection_valid.csv```

You can also use this script to generate the submission files on the test set: ```python inference.py --images_csv dataset/test_images.csv```. Don't forget to generate the test_images.csv before running inference (see the Preprocessing section).

Finally, to compute the F1-score, use the ```evaluator.py``` file, specifying the ```ground_truth_path``` and ```submission_file_path```. You should take into account that the number of images of both files should match, so if you generated predictions for the top-100 subset, your ```ground_truth_path``` should be ```dataset/concept_detection_valid_top100.csv```, while if you generated predictions for the whole validation set, it should be ```dataset/concept_detection_valid.csv```.

### Retrieval

### Semantic


## Caption Prediction

### Baseline without concepts

To train the baseline Vision Encoder-Decoder Transformer model just run ```python captioning/baseline_without_concepts/train.py```.

To evaluate your trained model on the validation set run ```python captioning/baseline_without_concepts/generation <checkpoint_to_trained_model>```.

To compute the evaluation scores run ```python captioning/eval-coco.py val_preds.json```. (The val_preds.json file is generated in the previous step.)

### Modified OSCAR

To install the OSCAR package run ```pip install -e captioning/Oscar```.

To train the modified OSCAR model run ```python captioning/Oscar/oscar/run_captioning.py --do_train --model_name_or_path <path_to_pretrained_checkpoint>```. This pretrained checkpoint can be obtained directly from the OSCAR repo. In the paper we used the coco_captioning_base_xe.zip checkpoint from [here](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#image-captioning-on-coco). 

To evaluate your trained model run ```python captioning/Oscar/oscar/run_captioning.py --do_eval --eval_model_dir <path_to_trained_checkpoint>```.



