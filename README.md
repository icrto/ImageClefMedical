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
```    

## Concept Detection

### Multilabel baseline

### Retrieval baseline

### Semantic baseline


## Caption Prediction

Start by converting the dataset to COCO format by calling ```
python3 captioning/baseline_without_concepts/dataset.py ```. This script will convert the caption_prediction_train.csv and caption_prediction_valid.csv files into COCO format and save them as caption_prediction_train_coco.json and caption_prediction_valid_coco.json, respectively.


### Baseline without concepts

### Modified OSCAR




