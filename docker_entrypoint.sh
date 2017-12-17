#!/usr/bin/env bash

cd /src
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
