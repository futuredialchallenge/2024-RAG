# SereTOD Track2: RAG-based Dialog Systems
This repository contains the README for Futuredial-RAG Track2.

Most existing domain-specific dialog systems do not carefully model the availble knowledge. Our systems require retrieval from multiple knowlege sources, including: a global knowledge base (global KB) covering all product information, a FAQ base (FAQ) covering recent asked questions and answers, a uniquelocal knowledge base for each dialog containing user specific knowledge (local KB) covering all product information. Compared with previous work, the task in Track2 has two main characteristics:
* There are multiple knowledge sources for the dialog system to search from.
* Multiple pieces of knowledge may be useful for response generation given the context.
*  Only a proportion of the dialogs is annotated with intents and local KBs. The teams are encouraged to utilize a mix of labeled and unlabeled dialogs to build a TOD system.

# Task Definition
The basic task for the TOD system is, for each dialog turn, given the dialog history, the user utterance, the local KB, and the global KB (including FAQ lists), to predict the useful knowledge pieces from the knowledge base and then generate an appropriate response. 
For every labeled dialog, useful knowledge pieces may be predicted by the retrieval model in Track 1, or the dialog system can predict the useful knowledge pieces itself.
For unlabeled dialogs, there are no knowledge base annotations.

# Baseline 
The folder provides a baseline for the retrieval system.  

### Setup
First, install all the requirements:
```Shell
pip install -r requirements.txt 
```

Then, use the following script to train the generation model:
```Shell
bash Track2/train.sh
```

Then, use the following script to test the generation model with oracle knowledge base (Note: this test mode will not be considered for the final submission):
```Shell
bash Track2/test.sh
```

Then, use the following script to test the generation model with a trained retriever:
```Shell
bash Track2/test_with_ret.sh
```
# Evaluation
In order to measure the performance of TOD systems, both automatic evaluation and human evaluation will be conducted. 
For automatic evaluation, metrics include Inform rate, Bertscore and BLEU score. Inform rate is the percentage of useful knowledge contained in the generated responses. BLEU score evaluates the fluency of generated responses. Bertscore evalutes the similarity between the generated responses and the ground-truth responses.

We will provide the following scripts and tools for the participants: 1) A baseline system; 2) Evaluation scripts to calculate the corpus-based metrics.

# Submission Format

# Data and Baseline

