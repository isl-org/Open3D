#!/bin/env bash

for f in Basic/*.ipynb; do jupyter nbconvert --to notebook --execute $f; done
for f in Advanced/*.ipynb; do jupyter nbconvert --to notebook --execute $f; done
