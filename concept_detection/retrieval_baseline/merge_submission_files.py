
import pandas as pd, numpy as np

multilabel_filename = './Submissions/multilabel.csv'
#multilabel_csv = pd.read_csv(multilabel_filename, header=0, sep="|")
#multilabel_images = multilabel_csv[:]['ID']
#multilabel_concepts = np.array(multilabel_csv.drop(['ID'], axis=1))
multilabel_csv = pd.read_csv(multilabel_filename, header=None, sep="|")
multilabel_images = np.array(multilabel_csv.iloc[:, 0])
multilabel_concepts = np.array(multilabel_csv.iloc[:, 1])
ml_index = np.argsort(multilabel_images)
multilabel_images = multilabel_images[ml_index]
multilabel_concepts = multilabel_concepts[ml_index]

retrieval_filename = './Submissions/retrieval.csv'
#retrieval_csv = pd.read_csv(retrieval_filename, header=0, sep="|")
#retrieval_images = retrieval_csv[:]['ID']
#retrieval_concepts = np.array(retrieval_csv.drop(['ID'], axis=1))
retrieval_csv = pd.read_csv(retrieval_filename, header=None, sep="|")
retrieval_images = np.array(retrieval_csv.iloc[:, 0])
retrieval_concepts = np.array(retrieval_csv.iloc[:, 1])
#ml_index = np.argsort(retrieval_images)
#retrieval_images = retrieval_images[ml_index]
#retrieval_concepts = retrieval_concepts[ml_index]

final_concepts_OR = []
final_concepts_NAN = []
nan_count = 0
for idx in range(len(multilabel_images)):
    if multilabel_images[idx] != retrieval_images[idx]:
        print('Different image order')
        exit(0)
    if multilabel_concepts[idx] != multilabel_concepts[idx]:
        nan_count += 1
        concepts = retrieval_concepts[idx].split(';')
        final_concepts_NAN.append(retrieval_concepts[idx])
    else:
        concepts = np.concatenate((multilabel_concepts[idx].split(';'), retrieval_concepts[idx].split(';')))
        final_concepts_NAN.append(multilabel_concepts[idx])
    concepts = np.unique(concepts)
    final_concepts_OR.append(';'.join(concepts))

eval_set = dict()
eval_set["ID"] = retrieval_images
eval_set["cuis"] = final_concepts_NAN

evaluation_df = pd.DataFrame(data=eval_set)
evaluation_df.to_csv('./merged_results_NAN.csv', sep='|', index=False, header=False)

print('NAN count: ' + str(nan_count))