# Lab 1 - Generative AI Use Case: Dialog summarization

In this use case, we will be generating a summary of a dialog with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face

Instructions available [here](https://www.coursera.org/learn/generative-ai-with-llms/gradedLti/loNJu/lab-1-generative-ai-use-case-summarize-dialogue)

This lab leverages AWS Sagemaker
Data can be copied into Sagemaker Studio notebok via terminal:
`aws s3 cp --recursive s3://dlai-generative-ai/labs/w1-549876/ ./`

The Notebook is available here:
 - [Lab_1_summarize_dialogue_executed.ipynb](Lab_1_summarize_dialogue_executed.ipynb)
 - [Lab_1_summar… - JupyterLab - executed.pdf](Lab_1_summarize_dialogue_executed.pdf)

To note to run it you will need a `ml.m5/2xlarge` instance type, and `python3` as kernel.

This notebook makes use of the below packages:
 - torch
 - torchdata
 - hugging face transformers => AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
 - hugging face datasets => load_dataset


And it uses Flan T5: `model_name='google/flan-t5-base'`
