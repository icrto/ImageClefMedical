# ImageClefMedical

This is the official repository for the VCMI's Team submission to [ImageClefmedical Caption 2022](https://www.imageclef.org/2022/medical/caption).

You can find the paper [here](http://ceur-ws.org/Vol-3180/paper-116.pdf).

For more information please contact isabel.riotorto@inesctec.pt.

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

Start by running ```python3 preprocessing/get_topconcepts.py``` to obtain the train_top100.csv, valid_top100.csv and top100_concepts.csv files. top100_concepts.csv corresponds to the concepts.csv file, but filtered to contain only the top-K most-frequent concepts. The train/valid_top100.csv correspond to the concept_detection_train/valid.csv files, respectively, but the concepts not present in the top-K are removed (images that end up without any valid concept are also removed).

## Concept Detection

### Multilabel baseline

### Retrieval baseline

### Semantic baseline


## Caption Prediction

Start by converting the dataset to COCO format by calling ```
python3 captioning/baseline_without_concepts/dataset.py ```. This script will convert the caption_prediction_train.csv and caption_prediction_valid.csv files into COCO format and save them as caption_prediction_train_coco.json and caption_prediction_valid_coco.json, respectively.


### Baseline without concepts

To train the baseline Vision Encoder-Decoder Transformer model, just run ```python3 captioning/baseline_without_concepts/train.py```

### Modified OSCAR




