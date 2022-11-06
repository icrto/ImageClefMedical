# ImageClefMedical

This is the official repository for the VCMI's Team submission to [ImageClefmedical Caption 2022](https://www.imageclef.org/2022/medical/caption).

You can find the paper [here](http://ceur-ws.org/Vol-3180/paper-116.pdf).

For more information please contact isabel.riotorto@inesctec.pt.

## Requirements

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

To train the baseline Vision Encoder-Decoder Transformer model, just run ```python3 captioning/baseline_without_concepts/train.py```.

### Modified OSCAR

To install the OSCAR package run ```pip install -e captioning/Oscar```.
To train the modified OSCAR model run ```python3 captioning/Oscar/oscar/run_captioning.py --do_train```.




