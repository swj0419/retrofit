# robust_emb
Code of Retrofitting Contextualized Word Embeddings with Paraphrases. 

## Install
Make sure your local environment has the following installed:

    Python3
    tensorflow >= 1.5.0
    h5py==2.8.0
    
Install the dependents using:

    pip install -r requirements.txt

## Run the experiments
To run the experiments, use:

    python ./run/run.py --data_path data.pk --options_file elmo_option_file 
    --weight_file elmo_weight_file --vocab_file elmo_vocabulary_file --token_emb_file elmo_token_embedding_file
   

## Data
Dataset used in experiements:

Microsoft paraphrase dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52398

Quora paraphrase dataset: https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs

Please use src/data.py to generate data pickle.

TBC
