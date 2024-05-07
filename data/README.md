# Mobile Customer-Service Dialog Dataset 2 (MobileCS2)

This directory contains examples for the official dataset for [SereTOD Challenge](../README.md).
The evaluation data will be released later.

## Data Description
An important feature for SereTOD challenge is that we release around 6,000 dialogs (in Chinese), which come from real-world dialog transcripts between real users and
customer-service staffs from China Mobile, with privacy information anonymized. 
We call this dataset as **MobileCS2** (mobile customer-service) dialog dataset, which differs from existing RAG datasets in both **nature** and **data-modelling** significantly.
To the best of our knowledge, MobileCS2 is not only a unique publicly available RAG-based dialog dataset, but also consists of real-life data (namely collected in real-world scenarios).

A schema is provided, based on which 3K dialogs are labeled by crowdsourcing. The remaining 3K dialogs are unlabeled.
The teams are required to use this mix of labeled and unlabeled data to train information retrieval models (Track 1), which could provide a knowledge base query results for Track 2, and train RAG-based dialog systems (Track 2), which could work as customer-service bots.
We put aside 413 dialogs as evaluation data. More details can be found in [Challenge Description](http://seretod.org/SereTOD_Challenge_Description_v2.0.pdf) .

## Data Format
We provide some dialog examples in [example.json](example.json), consisting of 100 dialogs.
The entire MobileCS dataset is provided to registered teams.

The dataset contains a list of instances, each of which is a dialog between a user and a customer-service staff. Each dialog is a list of turns. Each turn includes the following objects:
* Speaker ID: the speaker of the dialogue, such as "[SPEAKER 1]" and "[SPEAKER 2]"
* Intents: the intent of each speaker in this turn. "用户意图" represents the user intent, "客服意图" represents the system intent
* Information: including the entities and triples mentioned in this turn
