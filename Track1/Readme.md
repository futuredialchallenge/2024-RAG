# Track1: Information retrieval based on knowledge bases and dialog context 
In a RAG-based dialog system, the system needs to retrieval from knowledge base to get relevant knowledge. Given a mix of labeled and unlabeled dialog transcripts, Track 1 examines the task of training information retrieval models to retrieve knowledge from the knowledge base for each dialog, which will be needed in training dialog systems in Track 2.    

# Evaluation  
Given a dialog in testing, the trained information retrieval model is used to retrieve the knowledge pieces. We will evaluate and rank the submitted models by the retrieval model performance on test set. The evaluation metrics are Recall@1, Recall@5, and Recall@20.  

We will provide the following scripts and tools for the participants: 1) Baseline models for both sub-tasks; 2) Evaluation scripts to calculate the metrics.

# Baseline 
The folder provides a baseline for the retrieval system.  

### Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```

Then, generate the knowledge base and the retrieval training files from the training dataset using the following code:
```Shell
python process_data.py
python reader.py
```

Then, use the following script to train the retrieval model:
```Shell
python retrieve_kb.py
```

Note, you may need to download the pretrained model bge-large-zh-v1.5 to the corresponding place to initialize the retriever model. 
### Submission Format

The final submission should contain a json file as follows. 
```Json
[
    { // a doc
        "id": "2aa131d5143bddb3772f595292987780", // dial id,
        "turn_num": "0", // the turn number of the dial,
        "top20_id": [ 1,2,3,……] // ranked top 20 doc index, top 5, top1 can be directly gotten by [:5] and [:1]
    },
……   
]
```

### Evaluation and Results
We use recall as the basic metric. The results are:

| R@1  | R@5 | R@20 |
| ---: | :--- |:--- |
|0.300 | 0.478|0.606|

The results are relatively low, which indicates the task is challenging and needs more powerful models. 
