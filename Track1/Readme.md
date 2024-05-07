# Task    
In a RAG-based dialog system, the system needs to retrieval from knowledge base to get relevant knowledge. Given a mix of labeled and unlabeled dialog transcripts, Track 1 examines the task of training information retrieval models to retrieve knowledge from the knowledge base for each dialog, which will be needed in training TOD systems in Track 2.    
# Evaluation  
Given a dialog in testing, the trained information retrieval model is used to retrieve the knowledge pieces together with slot values. We will evaluate and rank the submitted models by the extraction performance on test set. The evaluation metrics are Precision, Recall and F1.  

The average F1 scores of entity extraction and slot filling will be the ranking basis on leaderboard. We will provide the following scripts and tools for the participants: 1) Baseline models for both sub-tasks; 2) Evaluation scripts to calculate the metrics.

# Baseline 
The folder provides a baseline for the retrieval system.  

### Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```

Then, use the following script to train the retrieval model:
```Shell
python retrieve_kb.py
```

### Submission Format


### Evaluation and Results
We use recall as the basic metric. The results are:

| R@1  | R@5 | R@5 |
| ---: | :--- |:--- |
|0.300 | 0.478|0.478|0.606|

The results are relatively low, which indicates the task is challenging and needs more powerful models. 
