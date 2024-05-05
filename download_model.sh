#!/bin/bash
apt install curl
# Export your Kaggle username and API key
export KAGGLE_USERNAME=osgoodou
export KAGGLE_KEY=4fa7dca916e227a68b7d83538100f593

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o /workspace/algorithm/models/model.tar.gz\
  https://www.kaggle.com/api/v1/models/google/gemma/pyTorch/2b/2/download