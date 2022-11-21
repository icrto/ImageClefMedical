import csv
from sklearn.metrics import f1_score

# IMAGECLEF 2022 CAPTION - CONCEPT DETECTION
class AIcrowdEvaluator:

  def __init__(self, ground_truth_path, **kwargs):
    """
    This is the AIcrowd evaluator class which will be used for the evaluation.
    Please note that the class name should be `AIcrowdEvaluator`
    `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
    """
    self.ground_truth_path = ground_truth_path
    # Ground truth dict => gt[image_id] = tuple of concepts
    self.gt = self.load_gt()
    

  def _evaluate(self, client_payload, _context={}):
    """
    This is the only method that will be called by the framework
    returns a _result_object that can contain up to 2 different scores
    `client_payload["submission_file_path"]` will hold the path of the submission file
    """
    print("evaluate...")
    # Load submission file path
    submission_file_path = client_payload["submission_file_path"]
    # Load preditctions and validate format
    predictions = self.load_predictions(submission_file_path)

    score = self.compute_primary_score(predictions)
    score_secondary = self.compute_secondary_score(predictions)

    _result_object = {
        "score": score,
        "score_secondary": score_secondary
    }
    
    assert "score" in _result_object
    assert "score_secondary" in _result_object

    return _result_object


  def load_gt(self):
    """
    Load and return groundtruth data
    """
    print("loading ground truth...")

    gt = {}
    with open(self.ground_truth_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader, None) # ignore header
        for row in reader: 
            # We have an ID and a set of concepts (possibly empty)
            image_id = row[0]
            concepts = tuple()
            if len(row) > 1:
                concepts = tuple(concept.strip() for concept in row[1].split(";"))
                gt[image_id] = concepts
            else:
                raise Exception("Answer file format is wrong. Organizer should check the answer file")
    return gt


  def load_predictions(self, submission_file_path):
    """
    Load and return a predictions object (dictionary) that contains the submitted data that will be used in the _evaluate method
    Validation of the runfile format has to be handled here. simply throw an Exception if there is a validation error.
    """
    print("load predictions...")
    predictions = {}
    image_ids_gt = tuple(self.gt.keys())
    max_num_concepts = 100
    with open(submission_file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
        lineCnt = 0
        occured_images = []
        for row in reader:
            lineCnt += 1
            #Token length not 1 and not 2 => Error
            if not 1 <= len(row) <=2:
                self.raise_exception("Wrong format. Each line must at least consist of an image ID, optionally followed"+
                     "by a vertical line (|) and 1 or more concepts separated my a semicolon ({}).", lineCnt,
                     "<image_id><vertical line><concept_1>;<concept_2>;<concept_3>;<concept_n>")

            image_id = row[0]
            # Image ID does not exist in testset => Error
            if image_id not in image_ids_gt:
                self.raise_exception("Image ID '{}' in submission file does not exist in testset.",lineCnt, image_id)

            occured_images = tuple(predictions.keys())
            # image id occured at least twice in file => Error
            if image_id in occured_images:
                self.raise_exception("Image ID '{}' was specified more than once in submission file.", lineCnt,
                    image_id)

            concepts = tuple()
            # more than max num concepts for image => Error
            if len(row) > 1:
                concepts = tuple(concept.strip() for concept in row[1].split(";"))
                if len(concepts) > max_num_concepts:
                    self.raise_exception("Too Many concepts specified. There must be between 0 and {} concepts per image.", lineCnt,
                        max_num_concepts)

            # concept(s) specified more than once for an image => Error
            if len(concepts) != len(set(concepts)):
                self.raise_exception("Same concept was specified more than once for image ID '{}'.", lineCnt, image_id)

            predictions[image_id] = concepts


        # In case not all images from the testset are contained in the file => Error
        if(len(predictions) != len (image_ids_gt)):
            self.raise_exception("Number of image IDs in submission file not equal to number of image IDs in testset.",lineCnt)

    return predictions

  def raise_exception(self, message, record_count, *args):
    raise Exception(message.format(*args)+" Error occured at line nbr {}.".format(record_count))


  def compute_primary_score(self, predictions):
    """
    Compute and return the primary score
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Valiation should be handled in the load_predictions method
    """
    print("compute primary score...")
    # Define max score and current score
    max_score = len(self.gt) #nbr images
    current_score = 0

    # Evaluate each predicted concept list against the ground truth
    for image_id in predictions:

        predicted_concepts = tuple(con.upper() for con in predictions[image_id])
        gt_concepts = tuple(con.upper() for con in self.gt[image_id])

        # Manage empty GT concepts (ignore in evaluation)
        if len(gt_concepts) == 0:
            max_score -= 1 #lower max score
        # Normal evaluation
        else:
            # Global set of concepts
            all_concepts = sorted(list(set(gt_concepts + predicted_concepts)))

            # Calculate F1 score for the current concepts
            y_true = [int(concept in gt_concepts) for concept in all_concepts]
            y_pred = [int(concept in predicted_concepts) for concept in all_concepts]

            f1score = f1_score(y_true, y_pred, average='binary')

            # Increase calculated score
            current_score += f1score

    return current_score/max_score



  def compute_secondary_score(self, predictions):
    """
    Compute and return the secondary score
    Ignore or remove this method if you do not have a secondary score to provide
    `predictions` : valid predictions in correct format
    NO VALIDATION OF THE RUNFILE SHOULD BE IMPLEMENTED HERE
    Valiation should be handled in the load_predictions method
    """
    print("compute secondary score...")

    return 0.0


    #PUT AUXILIARY METHODS BELOW
    #...
    #...


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(
        description="Arguments to run the script.")

  # Processing parameters
  parser.add_argument("--ground_truth_path",
                      type=str, required=True)
  parser.add_argument("--submission_file_path",
                      type=str, required=True)
  args = parser.parse_args()

  _client_payload = {}
  _client_payload["submission_file_path"] = args.submission_file_path
  
  # Instaiate a dummy context
  _context = {}

  # Instantiate an evaluator
  aicrowd_evaluator = AIcrowdEvaluator(args.ground_truth_path)
  
  # Evaluate
  result = aicrowd_evaluator._evaluate(_client_payload, _context)
  print(result)
