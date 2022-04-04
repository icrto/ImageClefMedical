import sys, argparse, string
import csv
import warnings

from sklearn.metrics import f1_score

def main(argv):

    # Hide warnings
    warnings.filterwarnings('ignore')

    # Concept stats
    min_concepts = sys.maxsize
    max_concepts = 0
    total_concepts = 0
    concepts_distrib = {}

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_file', help='path to the candidate file to evaluate')
    parser.add_argument('gt_file', help='path to the ground truth file')
    args = parser.parse_args()

    # Read files
    print('Input parameters\n********************************')

    print('Candidate file is "' + args.candidate_file + '"')
    candidate_pairs = readfile(args.candidate_file)

    print('Ground Truth file is "' + args.gt_file + '"')
    gt_pairs = readfile(args.gt_file)

    # Define max score and current score
    max_score = len(gt_pairs)
    current_score = 0

    # Check there are the same number of pairs between candidate and ground truth
    if len(candidate_pairs) != len(gt_pairs):
        print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
        exit(1)

    # Evaluate each candidate concept list against the ground truth
    print('Processing concept sets...\n********************************')

    i = 0
    for image_key in candidate_pairs:

        # Get candidate and GT concepts
        candidate_concepts = candidate_pairs[image_key].upper()
        gt_concepts = gt_pairs[image_key].upper()

        # Split concept string into concept array
        # Manage empty concept lists
        if gt_concepts.strip() == '':
            gt_concepts = []
        else:
            gt_concepts = gt_concepts.split(';')

        if candidate_concepts.strip() == '':
            candidate_concepts = []
        else:
            candidate_concepts = candidate_concepts.split(';')

        # Manage empty GT concepts (ignore in evaluation)
        if len(gt_concepts) == 0:
            max_score -= 1
        # Normal evaluation
        else:
            # Concepts stats
            total_concepts += len(gt_concepts)

            # Global set of concepts
            all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))

            # Calculate F1 score for the current concepts
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in candidate_concepts) for concept in all_concepts]

            f1score = f1_score(y_true, y_pred, average='binary')

            # Increase calculated score
            current_score += f1score

        # Concepts stats
        nb_concepts = str(len(gt_concepts))
        if nb_concepts not in concepts_distrib:
            concepts_distrib[nb_concepts] = 1
        else:
            concepts_distrib[nb_concepts] += 1

        if len(gt_concepts) > max_concepts:
            max_concepts = len(gt_concepts)

        if len(gt_concepts) < min_concepts:
            min_concepts = len(gt_concepts)

        # Progress display
        i += 1
        if i % 1000 == 0:
            print(i, '/', len(gt_pairs), ' concept sets processed...')

    # Print stats
    print('Concept statistics\n********************************')
    print('Number of concepts distribution')
    print_dict_sorted_num(concepts_distrib)
    print('Least concepts in set :', min_concepts)
    print('Most concepts in set :', max_concepts)
    print('Average concepts in set :', total_concepts / len(candidate_pairs))

    # Print evaluation result
    print('Final result\n********************************')
    print('Obtained score :', current_score, '/', max_score)
    print('Mean score over all concept sets :', current_score / max_score)


# Read a Tab-separated ImageID - Caption pair file
def readfile(path):
    try:
        pairs = {}
        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                # We have an ID and a set of concepts (possibly empty)
                if len(row) == 2:
                    pairs[row[0]] = row[1]
                # We only have an ID
                elif len(row) == 1:
                    pairs[row[0]] = ''
                else:
                    print('File format is wrong, please check your run file')
                    exit(1)

        return pairs
    except FileNotFoundError:
        print('File "' + path + '" not found! Please check the path!')
        exit(1)


# Print 1-level key-value dictionary, sorted (with numeric key)
def print_dict_sorted_num(obj):
    keylist = [int(x) for x in list(obj.keys())]
    keylist.sort()
    for key in keylist:
        print(key, ':', obj[str(key)])

# Main
if __name__ == '__main__':
    main(sys.argv[1:])
