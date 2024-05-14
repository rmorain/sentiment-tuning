# Black Box Controllable Text Generation

## Installation
1. `pip install -r requirements.txt` or `conda env create -f environment.yml`

## Sentiment control experiment
To train and evaluate the sentiment control model, 
1. Run `rlhf/rlhf.sh`

To evaluate a trained model,
1. Modify `rlhf/run_test.py` to include a path to the model
2. Run `rlhf/run_test.sh` 

## Toxicity avoidance experiment
To train a detoxification model, 
1. Run `rlhf/toxicity.sh`

To evaluate a trained model on the Perspective API
1. Add your Perspective API key to `rlhf/constants.py`
2. Modify `rlhf/perspective_api.py` to include a path to the trained model
3. Run `python rlfh/perspective_api.py`
