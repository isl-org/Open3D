#!/bin/env bash

for f in Basic/*.ipynb; do 
    jupyter nbconvert --ExecutePreprocessor.timeout=6000 --to notebook --inplace --execute $f; 
done

for f in Advanced/*.ipynb; do
    jupyter nbconvert --ExecutePreprocessor.timeout=6000 --to notebook --inplace --execute $f; 
done
