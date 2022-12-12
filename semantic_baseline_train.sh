#!/bin/bash

array=("Body Part, Organ, or Organ Component" "Spatial Concept" "Finding" "Pathologic Function" "Qualitative Concept" "Diagnostic Procedure" "Body Location or Region" "Functional Concept" "Miscellaneous Concepts")
for i in "${array[@]}"
do
	python3 concept_detection/semantic/train.py --semantic_type "$i" --outdir results/semantic
done