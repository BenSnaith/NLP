# NLP
Natural Language Processing Coursework; Option 1: Emotional Classification of Tweets.

All aspects of this repository were created by Ben Snaith (230106507).

NOTE: IN THE CASE OF A SEVERE ERROR WITH SUBMISSION, THE WORKING SOURCE CODE CAN BE FOUND [HERE](https://github.com/BenSnaith/NLP)

## Setup
```shell
python3 -m venv venv
# Windows
venv\Scripts\activate
# Unix
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```shell
python3 src/main.py
```

Once ran outputs will be found in `results/`

## Customisation

Configration can be set by using the dictionaries at the top of `src/main.py`:

```py
TEST_RATIO = 0.1 # 10% testing, 90% train

RANDOM_SEED = 230106507 # set seed for reproducability

TFIDF_CONFIG = {
    'max_features': 5000, # vocabulary size
    'ngram_range': (1, 2), # use unigrams and bigrams
    'min_df': 2,
    'max_iter': 1000, # max iterations
}

TRANSFORMER_CONFIG = { # applies to both GPT2 and BERT
    'batch_size': 16, # reduce if out of memory
    'learning_rate': 5e-5, # learning rate
    'num_epochs': 3, # number of training epochs
    'max_length': 128, # max sequence length
}
```

## Dataset

Dataset: `data/raw/twitter_emotion_data.csv`

## Requirements

[Python 3.8+](https://www.python.org/)
(if installing for windows ensure you add python to `$PATH` and install pip)
