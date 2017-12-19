#!/usr/bin/env bash

echo "Create data folder..."

DATA_DIR=/src/data
mkdir -p $DATA_DIR

echo "Word2vec data"

# W2V_WEIGHTS_FILE in Dockerfile ENV
W2V_WEIGHTS_ARCHIVE=${W2V_WEIGHTS_FILE}.gz

echo "Download..."
wget http://rusvectores.org/static/models/rusvectores2/ruwikiruscorpora_rusvectores2.bin.gz --no-verbose -O $W2V_WEIGHTS_ARCHIVE 

echo "Extract..."
gunzip -c $W2V_WEIGHTS_ARCHIVE > ${DATA_DIR}/${W2V_WEIGHTS_FILE}

echo "Clear..."
rm $W2V_WEIGHTS_ARCHIVE

echo "Polyglot data"

echo "Download..."
polyglot download embeddings2.ru
polyglot download ner2.ru

echo "Done!"
