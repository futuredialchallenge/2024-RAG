This is the code repository for [FutureDial-RAG Challenge](http://futuredial.org/), co-located with [IEEE SLT 2024](https://2024.ieeeslt.org)!

# Introduction

Developing intelligent dialog systems has been one of the longest running goals in AI. Significant progress has been made in building dialog systems with deep learning methods and more recently with large language models. However, there are still full of challenges toward building future dialog systems. Following the success of the 1st FutureDial challenge (SereTOD), co-located with EMNLP 2022, we organize the 2nd FutureDial challenge, to further promote the study of how to empower dialog systems with RAG (Retrieval augmented generation). We will release a new dialog dataset, MobileCS2, in both Chinese and English versions. The dataset originates from real-life customer-service logs from China Mobile.

The task includes two tracks:

* Information retrieval based on knowledge bases and dialog context (Track 1)
* Dialog systems with retrieval augmented generation (Track 2)

**Organizers:** Junlan Feng, Zhijian Ou, Yi Huang, Si Chen, Yucheng Cai

# Important Dates (AOE)
| Date  | Item  |
| ---: | :--- |
|April 9, 2024 | Registration opening for the challenge|
|April 20, 2024 | Training data release|
|May 9, 2024 | Registration deadline for the challenge|
|June 10, 2024 | Entry submission deadline |
|June 20, 2024| Evaluation results announced|
|June 20, 2024|SLT paper submission deadline|
|June 27, 2024|SLT paper update deadline|
|August 30, 2024|Notification of paper acceptance|
|December 2-5, 2024	|SLT workshop|

# Important Links

If you publish experimental results with the MobileCS2 dataset and refer to the baseline models, please cite this challenge description paper:


#Rules

* The challenge website is http://futuredial.org/ . Teams should submit the registration form to FutureDialRAG@gmail.com, which will be reviewed by the organizers.
* Teams are required to sign an Agreement for Challenge Participation and Data Usage. Data will be provided to approved teams.
* For teams that participate in Track 1, the scores will be ranked according to the performance for Track 1. The teams can choose to participate only in Track 1.
* For teams that participate in Track 2, they can use the baseline system provided by the organizers or use the system developed by themselves for Track 1. The ranking is based on the performance for Track 2.
* Participants are allowed to use any external datasets, resources or pre-trained models which are publicly available.
* Participants are NOT allowed to do any manual examination or modification of the test data.
* In publishing the results, all teams will be identified as team IDs (e.g. team1, team2, etc). The organizers will verbally indicate the identities of all teams at the Challenge for communicating results.
* For each track, three teams with top performances will be recognized with prizes. The prizes will be awarded at the Workshop.
* Participants may identify their own team label (e.g. team5) and report their own result, in publications or presentations, if they desire.
  
# Submission Guidelines

* Each team needs to submit a package via email to FutureDialRAG@gmail.com before the Entry Submission Deadline. The submission package should contain the system executable with the model. All dependencies must be contained in the submission package.
  - For track 2, system running is not only for corpus-based automatic evaluation, but also for human evaluation.That is, the submission system for Track 2 should provide an interface, through which real users interact with those systems.
* The submission should provide clear documentation for running the system. Direct running the executable without any arguments should output the result file with the required format. See Track1 README and Track2 README for the formats.
* The submitted system could be, but not limited to be, encapsulated in a Docker image, as long as the above requirements are satisfied.
* The submission should provide a System Description Paper. Teams are also encouraged to submit papers to SLT2024. See important dates and instructions in Call for Papers.
* Before the Entry Submission Deadline, each team can submit for multiple times for each track. The last entry from each team will be used for the evaluation.
