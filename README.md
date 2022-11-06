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

Run ```python3 preprocessing/get_topconcepts.py```. This script will create the following files:
1. concepts_top100.csv
2. concept_detection_train_top100.csv e concept_detection_valid_top100.csv
3. caption_prediction_train_top100.csv e caption_prediction_valid_top100.csv

The concepts_top100.csv corresponds to the concepts.csv file, but filtered to contain only the top-K most-frequent concepts. The other files correspond to their original counterparts but with the concepts not present in the top-K removed (images that end up without any valid concept are also removed).

### Conversion to COCO format

Run ```
python3 preprocessing/convert_to_coco.py <your_file_here>```. This script will convert the given file into COCO format and save it as <your_file_here>_coco.json.

You should call this script for both caption_prediction_train.csv and caption_prediction_valid.csv files (and/or their top-K versions).

## Concept Detection

### Multilabel baseline

### Retrieval baseline

### Semantic baseline


## Caption Prediction

### Baseline without concepts

To train the baseline Vision Encoder-Decoder Transformer model just run ```python3 captioning/baseline_without_concepts/train.py```.

To evaluate your trained model on the validation set run ```python3 captioning/baseline_without_concepts/generation <checkpoint_to_trained_model>```.

To compute the evaluation scores run ```python3 captioning/eval-coco.py <val_preds.json>```. (The val_preds.json file is generated in the previous step.)

### Modified OSCAR

To install the OSCAR package run ```pip install -e captioning/Oscar```.

To train the modified OSCAR model run ```python3 captioning/Oscar/oscar/run_captioning.py --do_train --model_name_or_path <path_to_pretrained_checkpoint>```. This pretrained checkpoint can be obtained directly from the OSCAR repo. In the paper we used the coco_captioning_base_xe.zip checkpoint from [here](https://github.com/microsoft/Oscar/blob/master/VinVL_MODEL_ZOO.md#image-captioning-on-coco). 

To evaluate your trained model run ```python3 captioning/Oscar/oscar/run_captioning.py --do_eval --eval_model_dir <path_to_trained_checkpoint```.



