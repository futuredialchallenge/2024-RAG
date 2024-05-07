This is the code repository for [FutureDial-RAG Challenge](http://futuredial.org/), co-located with [IEEE SLT 2024](https://2024.ieeeslt.org)!

# Introduction

Developing intelligent dialog systems has been one of the longest running goals in AI. Significant progress has been made in building dialog systems with deep learning methods and more recently with large language models. However, there are still full of challenges toward building future dialog systems. Following the success of the 1st FutureDial challenge ([SereTOD](http://seretod.org/)), co-located with EMNLP 2022, we organize the 2nd FutureDial challenge, to further promote the study of how to empower dialog systems with **RAG (Retrieval augmented generation)**. We will release a new dialog dataset, MobileCS2, in both *Chinese* and *English* versions. The dataset originates from real-life customer-service logs from China Mobile.

The task includes two tracks:

* Information retrieval based on knowledge bases and dialog context (Track 1)
* Dialog systems with retrieval augmented generation (Track 2)

**Organizers:** Junlan Feng, Zhijian Ou, Yi Huang, Si Chen, Yucheng Cai

# Important Dates (AOE)
| Date  | Item  |
| ---: | :--- |
|April 9, 2024 | Registration opening for the challenge|
|April 29, 2024 | Training data release (extended)|
|May 9, 2024 | Registration deadline for the challenge|
|June 10, 2024 | Entry submission deadline |
|June 20, 2024| Evaluation results announced|
|June 20, 2024|SLT paper submission deadline|
|June 27, 2024|SLT paper update deadline|
|August 30, 2024|Notification of paper acceptance|
|December 2-5, 2024	|SLT2024 workshop (in-person)|

# Challenge Rules

* The challenge website is http://futuredial.org/ . Teams should submit the registration form to FutureDialRAG@gmail.com, which will be reviewed by the organizers.
* Teams are required to sign an Agreement for Challenge Participation and Data Usage. Data will be provided to approved teams.
* For teams that participate in Track 1, the scores will be ranked according to the performance for Track 1. The teams can choose to participate only in Track 1.
* For teams that participate in Track 2, they can use the baseline system provided by the organizers or use the system developed by themselves for Track 1. The ranking is based on the performance for Track 2.
* Participants need to strictly follow the Submission Guidelines as described below. Participants are allowed to use any external datasets, resources or pre-trained models which are publicly available.
* The evaluation data will not released to the teams for their own evaluation. The organizers will run the submitted systems for evaluation. The evaluation data will be shared with the eligible teams after evaluation results are announced. Only teams who strictly follow the Submission Guidelines are viewed as eligible.
* In publishing the results, all teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the Challenge for communicating results. Participants may identify their own team label (e.g. team5) and report their own result, in publications or presentations, if they desire.
  
# Submission Guidelines

* Each team needs to submit a package via email to FutureDialRAG@gmail.com before the Entry Submission Deadline. The package should contain a clear README documentation for running the system over the evaluation data. The submitted system should be in one of the following two forms. In either form, the system's processing speed should be no less than 10 tokens per second.
  - The submission package contains the system executable with the model, for example, in a Docker image. All dependencies are contained in the submission package. The organizers run the system over a server with Nvidia A100*4 hardware, evaluate, and calculate the running time over the evaluation data.
  - The system is encapsulated as a callable web service. The organizers will run the script submitted by the team, call the web service to evaluate, and calculate the running time over the evaluation data.
* The submission should provide a System Description Document (SDD), introducing the submitted system. Teams are also encouraged to submit papers to SLT 2024. See important dates and instructions at SLT 2024 website [IEEE SLT 2024](https://2024.ieeeslt.org).
* Before the Entry Submission Deadline, each team can submit for multiple times for each track. The last entry from each team will be used for the evaluation.
