# robust_emb
Code of Retrofitting Contextualized Word Embeddings with Paraphrases. (In progress)

## Install
Make sure your python environment has the following installed:

    Python3
    tensorflow >= 1.5.0
    scikit-learn
    
Install the dependents using:

    pip install -r requirements.txt
    
## Run
To run the experiments, use:

    python ./run/run.py --options_file "model options" --token_emb_file "token embedding file" --weight_file "model weight file" --data_path "path to data"



## Dataset
Machine Translation Metrics Paraphrase Corpus: https://pan.webis.de/clef10/pan10-web/plagiarism-detection.html

MSRP: https://www.microsoft.com/en-us/download/details.aspx?id=52398

Quora Question Pairs: https://www.kaggle.com/quora/question-pairs-dataset