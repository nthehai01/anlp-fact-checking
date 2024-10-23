# Fact-Checking Complex Claims with Program-Guided Reasoning

## Overview of the Task
* Problem:
    * In the world of exponential information growth, the task of fact-checking has become increasingly important, especially in the context of social media.
    * For instance, consider the claim "Both James Cameron and the director of the film Interstellar were born in Canada". It may be challenging to verify this claim by simply looking up on the internet.

* What we need:
    * An automated system that can verify the veracity of the complex claims.
    * A system that should be both explainable and data efficient.

## Proposed Framework
* **Program-Guided Fact-Checking (ProgramFC)**, a two-step framework that mimic how a human fact-checker would verify a claim:
    * Step 1 - *Decomposition*: The claim is decomposed into a set of sub-tasks:
        * A *reasoning program*, a sequence of sub-tasks, is generated for an input claim.
        * The reasoning program is considered as a step-by-step guide for verifying the claim.
    * Step 2 - *Piece-by-piece Solving*: Each sub-task is solved independently by leveraging the off-the-shelf solvers and the results are combined to verify the claim:
        * The generated reasoning program is executed sequentially by the corresponding sub-task handlers.
        * These sub-tasks may be answering questions, verifying simple claims, or logical reasoning.

* Key points:
    * No training data needed.
    * Using LLMs with the in-context learning capability by providing some (claim, program) pairs.
    * Flexibility in choosing the solvers for the sub-tasks.
    * These sub-task solvers can retrieve information in open-book or closed-book settings.

* Evaluation:
    * The proposed framework is evaluated on the HOVER & FEVER datasets and it Outperforms seven few-shot fact-checking baselines on both datasets.
    * Becoming increasingly effective as the reasoning program becomes more complex.
    * Also robust even using the weak sub-task solvers.
    * The interpretability of the reasoning program is also evaluated through human evaluation and error analysis.  

## Framework Details
### Formulation
Given a claim $C$, a fact-checking model $F$ aims to predict a label $Y$ to evaluate the claim as TRUE or FALSE, based on a knowledge source $K$.

Three types of knowledge sources $K$:
* *Gold evidence*: $K$ is the condensed evidence that supports for verifying the claim.
* *Open-book setting*: $K$ is a very large knowledge source such as Wiki, the model should retrieve the relevant evidence before verifying the claim.
* *Closed-book setting*: $K = \emptyset$.

### Execution
**Program Generation**:
* $n$ sequentially ordered reasoning steps are generated for the input claim $C$ such that each step is an natural language instruction that can be passed to the sub-task solvers in the next step.
* The output of these reasoning steps can also be an argument for the next reasoning steps or simply a boolean value that can be used for the result aggregation.
* For example, in Figure 1, the argument in S3 is "{ANSWER_1} was born in Canada.", which refers to the return variable {ANSWER_1} from S2. When executing S3 becomes "Christopher Nolan was born in Canada".
* **CodeX** is used to generate the reasoning program in the Python-like format by providing it some exemplars.

**Program Execution**:
* Three types of Sub-Task Functions:
    * Question: Using FLAN-T5 to answer the question.
    * Verify: Also using FLAN-T5 to return a boolean label.
    * Predict: Simply performing a logical reasoning AND/OR/NOT operations to combine the results from the previous steps and produce the *predicted veracity label* for a given claim $C$.

## Experiments
### General Settings
#### Datasets: HOVER & FEVEROUS
* Tested on the dev set as the test set is not publicly available.
* Focus on complex claims that need multi-step reasoning.
* *HOVER* contains claims that require integration and reasoning over multiple Wikipedia articles.
* *FEVEROUS* focuses on fact-checking complex claims over unstructured and structured data, where each claim is annotated with evidence in the form of sentences and/or cells from tables in Wikipedia. Only claims requiring exclusively sentence evidence are selected for evaluation.
* For the evaluation in *open-book* setting, they use the corresponding Wikipedia corpus constructed for these two datasets as *K*.

#### Baselines
7 baselines in 3 categories:
* *Pretrained models:* BERT-FC & LisT5. 
* *FC/NLI fine-tuned models*: 
    * RoBERTa-NLI, fine-tuned RoBERTa-large on four NLI
datasets.
    * DeBERTaV3-NLI; fine-tuned DeBERTaV3 on 885,242 (claim, evidence, label) annotations from FEVER and four NLI datasets.
    * MULTIVERS, fine-tuned LongFormer on FEVER.
* *In-context learning models*: 
    * FLAN-T5 is directly used in the VERIFY module.
    * Codex, for few-shot fact-checking.

### Main Results
*Look at the Table 1 in the paper*, there are 3 observations:

1. More deeper claims means more effective **compared to others**.
    * On the HOVER dataset, ProgramFC (N=5) outperforms the baselines on average by 10.38%, 11.37%, and 14.77% on two-hop, three-hop, and four-hop claims, respectively.
    * Also on HOVER, F1 score of DeBERTaV3-NLI drops 21.7% while ProgramFC (N=5) only drops 11.7%.
2. Decomposition is more effective than direct prediction.
    * FLAN-T5 model (in ProgramFC) outperforms the baseline FLAN-T5 of directly verifying claims across *all datasets*.
    *  Especially evident when the required reasoning is complex.
3. Aggregating reasoning programs is helpful.
    * ProgramFC (N=5) improves the performance over using ProgramFC (N=1) by an average of 1.5%.

### Comparison of FLAN-T5 with and without ProgramFC using different model sizes
* In the gold evidence setting, *Table 4 in the paper*:
    * Program-guided reasoning is particularly effective when the model size is small.
    * Performance of the end-to-end FLAN-T5 model decreases significantly with decreasing model size.
    * Program-guided model using FLAN-T5-small as sub-task solvers can achieve comparable performance to the 137x larger FLAN-T5-XXL.
* In the open-domain setting, *Table 5 in the paper*:
    * Outperforms one-step retrieval on all datasets, especially on HOVER 4-hop.
    * It is because some information is only revealed during the reasoning process but not in the original claim.

## Limitations
* The claims in the HOVER and the FEVEROUS datasets mostly only require *explicit* multi-step reasoning. CodeX-based generator struggles when dealing with implicit complex claims requires a deeper understanding.
* Higher computational cost than the baseline, computational time that is ∼4-5x higher than for an end-to-end FLAN-T5.
