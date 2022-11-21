import pandas as pd

from Authentication import *
import requests
import json
import argparse
import os
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Add the arguments
# Data directory
parser.add_argument('--concepts_csv', type=str, default="dataset/concepts.csv", help="csv file with concepts.")

args = parser.parse_args()

apikey = # INSERT YOUR API KEY HERE
version = "current"
AuthClient = Authentication(apikey)

###################################
# get TGT for our session
###################################

tgt = AuthClient.gettgt()
uri = "https://uts-ws.nlm.nih.gov"

###################################
# Get CUI
###################################

df_concepts = pd.read_csv(args.concepts_csv, sep="\t", header=0)

all_concepts = df_concepts["concept"][:1]

dict_concept_semantic = dict()

for c in tqdm(all_concepts):

    # Concept ID
    identifier = c

    content_endpoint = "/rest/content/" + str(version) + "/CUI/" + str(identifier)

    ##ticket is the only parameter needed for this call - paging does not come into play because we're only asking for one Json object
    try:
        query = {'ticket': AuthClient.getst(tgt)}
        r = requests.get(uri + content_endpoint, params=query)
        r.encoding = 'utf-8'
        items = json.loads(r.text)
        jsonData = items["result"]

        ui = jsonData["ui"]

        jsonData["semanticTypes"]
        list_semantic = []
        for stys in jsonData["semanticTypes"]:
            list_semantic.append(stys['name'])

        dict_concept_semantic[ui] = list_semantic
    except:
        pass

fields = ['concept', 'semantic_types']

savepath = os.path.join(os.path.dirname(args.concepts_csv), 'semantic_types.csv')
print("Saving file to:", savepath)
with open(savepath, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for concept in dict_concept_semantic:
        writer.writerow({'concept': concept, 'semantic_types': " ".join(map(str, dict_concept_semantic[concept])).replace('"', '')})