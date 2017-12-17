#!/usr/bin/env bash

echo "Create data folder..."

DATA_DIR=/data
mkdir $DATA_DIR

echo "Polyglot data"

echo "Download..."
polyglot download embeddings2.ru
polyglot download ner2.ru

echo "Done!"
