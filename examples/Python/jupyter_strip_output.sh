#!/bin/env bash

for f in Basic/*.ipynb; do 
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $f; 
done

for f in Advanced/*.ipynb; do 
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $f;
done
