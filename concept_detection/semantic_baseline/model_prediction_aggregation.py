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
for subset in [test_path, validation_path]:

    # Get directory
    eval_dir = os.path.join(results_path, semantic_baseline_path, subset)

    # Read directory
    csv_files = [c for c in os.listdir(eval_dir) if not c.startswith('.')]
    csv_files = [c for c in csv_files if c not in ("test_agg.csv", "val_agg.csv")]

    assert len(csv_files) == 9, "Check your directory."

    # Pre-create image lists
    eval_data = dict()

    # Go through all the .CSV files
    for csv in csv_files:

        # Read .CSV
        if subset == "validation":
            df = pd.read_csv(os.path.join(eval_dir, csv), sep="\t")

        else:
            df = pd.read_csv(os.path.join(eval_dir, csv), sep="|", header=None)

        # Create image list
        for _, row in df.iterrows():
            if subset == "validation":
                if row["ID"] not in eval_data.keys():
                    eval_data[row["ID"]] = list()

            else:
                if row[0] not in eval_data.keys():
                    eval_data[row[0]] = list()

        # Append concepts (if different from 'None')
        for index, row in df.iterrows():

            if subset == "validation":
                eval_data[row["ID"]] += str(row["cuis"]).split(';')

                for i, c in enumerate(eval_data[row["ID"]]):
                    if c in ("None", "nan"):
                        eval_data[row["ID"]].pop(i)

                # Remove duplicates if needed (we don't know why this happens)
                eval_data[row["ID"]] = list(
                    dict.fromkeys(eval_data[row["ID"]]))

            else:
                eval_data[row[0]] += str(row[1]).split(';')

                for i, c in enumerate(eval_data[row[0]]):
                    if c in ("None", "nan"):
                        eval_data[row[0]].pop(i)

                # Remove duplicates if needed (we don't know why this happens)
                eval_data[row[0]] = list(dict.fromkeys(eval_data[row[0]]))

    # Process concept lists
    for key, value in eval_data.items():
        # Add the valid concepts
        predicted_concepts = ""
        for c in value:
            predicted_concepts += f"{c};"

        eval_data[key] = predicted_concepts[:-1]

    # Convert this data into a DataFrame
    df_dict = dict()
    df_dict["ID"] = list()
    df_dict["cuis"] = list()
    for key, value in eval_data.items():
        df_dict["ID"].append(key)
        df_dict["cuis"].append(value)

    # print(df_dict)
    evaluation_df = pd.DataFrame(data=df_dict)

    # Select correct subsets
    if subset == "validation":

        # Open validation ground-truth
        data_dir = os.path.join("data", "csv", "concepts")
        concept_dict_name_to_idx, concept_dict_idx_to_name, _ = get_concepts_dicts(
            data_dir=data_dir, concepts_csv="concepts.csv")
        # print(concept_dict_name_to_idx)

        # Open ground-truth data
        valid_df = pd.read_csv(os.path.join(
            data_dir, "concept_detection_valid.csv"), sep="\t")

        # Join DataFrames to obtain a DataFrame with Ground-Truth and Predicted values as columns
        predictions_df = evaluation_df.copy().merge(
            valid_df.copy(), on="ID", suffixes=('_pred', '_gt'))
        predictions_df.dropna()
        # print(predictions_df)

        # Generate label matrices
        # y_true
        y_true = np.zeros((len(predictions_df), len(concept_dict_name_to_idx)))
        # print(valid_df['cuis'].values)
        for index, row in predictions_df.iterrows():
            concepts = row['cuis_gt'].split(';')
            for c in concepts:
                y_true[index, concept_dict_name_to_idx.get(c)] = 1

        y_pred = np.zeros_like(y_true)
        for index, row in predictions_df.iterrows():
            concepts = row['cuis_pred'].split(';')
            for c in concepts:
                y_pred[index, concept_dict_name_to_idx.get(c)] = 1

        print(f"/////////// Final Evaluation Report ////////////")
        print(
            f"Exact Match Ratio: {metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
        print(f"Hamming loss: {metrics.hamming_loss(y_true, y_pred):.4f}")
        print(
            f"Recall: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(
            f"Precision: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(
            f"F1 Measure: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")

        # Top-100 Report
        sem_concepts_path = os.path.join(data_dir, "top100")
        concept_dict_name_to_idx, concept_dict_idx_to_name, _ = get_concepts_dicts(
            data_dir=sem_concepts_path, concepts_csv="new_top100_concepts_sem.csv")
        valid_csvpath = os.path.join(
            data_dir, "top100", "new_val_subset_top100_sem.csv")
        valid_df = pd.read_csv(valid_csvpath, sep="\t")

        # Join DataFrames to obtain a DataFrame with Ground-Truth and Predicted values as columns
        predictions_df = evaluation_df.copy().merge(
            valid_df.copy(), on="ID", suffixes=('_pred', '_gt'))
        predictions_df.dropna()
        # print(predictions_df)

        # Generate label matrices
        # y_true
        y_true = np.zeros((len(predictions_df), len(concept_dict_name_to_idx)))
        # print(valid_df['cuis'].values)
        for index, row in predictions_df.iterrows():
            concepts = row['cuis_gt'].split(';')
            for c in concepts:
                y_true[index, concept_dict_name_to_idx.get(c)] = 1

        y_pred = np.zeros_like(y_true)
        for index, row in predictions_df.iterrows():
            concepts = row['cuis_pred'].split(';')
            for c in concepts:
                y_pred[index, concept_dict_name_to_idx.get(c)] = 1

        print(f"/////////// Final Evaluation Report Top-100 ////////////")
        print(
            f"Exact Match Ratio: {metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):.4f}")
        print(f"Hamming loss: {metrics.hamming_loss(y_true, y_pred):.4f}")
        print(
            f"Recall: {metrics.precision_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(
            f"Precision: {metrics.recall_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")
        print(
            f"F1 Measure: {metrics.f1_score(y_true=y_true, y_pred=y_pred, average='samples'):.4f}")

        # Save .CSV of results
        evaluation_df.to_csv(os.path.join(
            eval_dir, "val_agg.csv"), sep="\t", index=False)

    else:
        evaluation_df.to_csv(os.path.join(
            eval_dir, "test_agg.csv"), sep="|", index=False, header=False)


print("Finished.")
