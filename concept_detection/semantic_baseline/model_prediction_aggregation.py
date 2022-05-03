# Imports
import os
import sys
import pandas as pd
import numpy as np

# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Sklearn Imports
from sklearn import metrics


# Project Imports
from aux_utils.aux_functions import get_concepts_dicts


# Results paths
results_path = "results"
semantic_baseline_path = "semantic_baseline"
validation_path = "validation"
test_path = "test"


# Iterate through subsets
for subset in [validation_path, test_path]:

    # Get directory
    eval_dir = os.path.join(results_path, semantic_baseline_path, subset)

    # Read directory
    csv_files = [c for c in os.listdir(eval_dir) if not c.startswith('.')]

    # Pre-create image lists
    eval_data = dict()
    eval_images = list()
    eval_concepts = list()

    # Go through all the .CSV files
    for csv in csv_files:
        
        # Read .CSV
        if subset == "validation":
            df = pd.read_csv(os.path.join(eval_dir, csv), sep="\t")
        
        else:
            df = pd.read_csv(os.path.join(eval_dir, csv), sep="|")
        
        # Convert .CSV into array
        df_values = df.values
        # print(df_values)

        # Create image list
        for _, row in df.iterrows():
            if subset == "validation":
                if row["ID"] not in eval_images:
                    eval_images.append(row["ID"])
                    eval_concepts.append(list())
            
            else:
                if row.iloc[0, 0] not in eval_images:
                    eval_images.append(row.iloc[0, 0])
                    eval_concepts.append(list())

        
        # print(eval_images)

        # Append concepts (if different from 'None')
        for index, row in df.iterrows():
            # print(value[1])
            if subset == "validation":
                pred_concepts = str(row["cuis"]).split(';')
            
            else:
                pred_concepts = str(row.iloc[0, 1]).split(';')
            
            for c in pred_concepts:
                if c not in ("None", "nan"):
                    eval_concepts[index].append(c)
        
        # print(eval_concepts)
    
    # print(eval_concepts)

    # Process concept lists
    for index, value in enumerate(eval_concepts):
        # Add the valid concepts
        predicted_concepts = ""
        for c in value:
            predicted_concepts += f"{c};"
        
        eval_concepts[index] = predicted_concepts[:-1]
    
    # print(eval_concepts)

    # Convert this data into a DataFrame
    eval_data["ID"] = eval_images
    eval_data["cuis"] = eval_concepts
    evaluation_df = pd.DataFrame(data=eval_data)

    if subset == "validation":
        
        # Open validation ground-truth
        data_dir = os.path.join("data", "csv", "concepts")
        concept_dict_name_to_idx, concept_dict_idx_to_name, _ = get_concepts_dicts(data_dir=data_dir, concepts_csv="concepts.csv")
        # print(concept_dict_name_to_idx)

        # Open ground-truth data
        valid_df = pd.read_csv(os.path.join(data_dir, "concept_detection_valid.csv"), sep="\t")

        # Generate label matrices
        # y_true
        y_true = np.zeros((len(eval_images), len(concept_dict_name_to_idx)))
        # print(valid_df['cuis'].values)
        index = 0
        for _, row in valid_df.iterrows():
            if row['ID'] in eval_images:
                    concepts = row['cuis'].split(';')
                    for c in concepts:
                        y_true[index, concept_dict_name_to_idx.get(c)] = 1
                    
                    index += 1
        

        y_pred = np.zeros_like(y_true)
        for index, set_concepts in enumerate(eval_concepts):
            concepts = set_concepts.split(';')
            for c in concepts:
                y_pred[index, concept_dict_name_to_idx.get(c)] = 1


        print(f"/////////// Final Evaluation Report ////////////")
        print(f"Exact Match Ratio: {metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
        print(f"Hamming loss: {metrics.hamming_loss(y_true, y_pred):.4f}")
        print(f"Recall: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(f"Precision: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(f"F1 Measure: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        

        # Save .CSV of results
        evaluation_df.to_csv(os.path.join(eval_dir, "val_agg.csv"), sep="\t", index=False)
    
    
    else:
        evaluation_df.to_csv(os.path.join(eval_dir, "test_agg.csv"), sep="|", index=False, header=False)


print("Finished.")
