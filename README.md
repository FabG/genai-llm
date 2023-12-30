# Gen AI LLM


This repo inclueds courses, code snippets from various sources to better learn and build an intuition about Large Language Models
I first enrolled in a course from Deeplearning.ai called [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) Course from AWS and Deeplearning.ai.
The main reason is my prior experience with Deeplearnin.ai as I gained a lot of knowledge from 2 prior courses from Andrew Ng and his team, namely [Machine Learning specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) and [Deep Learning specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)

<img src="images/generative-ai-with-llms-1.png" alt="course_llm" width="100"/>

### Courses - Generative AI with Large Language Model
 - Course 1 - Intro to LLMs and the generative AI project lifecycle
   - [1 - Transformer intro and arch](genai-with-llm-course/course1/1_transformer_intro_arch.md)
   - [2 - Generating Text with Transformers](genai-with-llm-course/course1/2_text_generation_transformer.md)
   - [3 - Attention is All you need (paper)](genai-with-llm-course/course1/3_attention_is_all_you_need.md)
   - [4 - Prompt Engineering](genai-with-llm-course/course1/4_prompt_engineering.md)
   - [5 - Generative Configuration and GenAI Project Lifecycle](genai-with-llm-course/course1/5_generative_configuration.md)
   - [Lab - Document Summarization with LLM Hugging Face FLAN T5](genai-with-llm-course/course1/lab_dialog_summarization.md)
 - Course 2 - LLM pre-training and scaling laws
   - [1 - Pre-Training large language models](genai-with-llm-course/course2/1_pre-training_llm.md)
   - [2 - Computational Challenges of training LLMs](genai-with-llm-course/course2/2_computational_challenges_training_llm.md)
   - [3 - Efficient multi-GPU compute strategies](genai-with-llm-course/course2/3_multi_gpu_strategies.md)
   - [4 - Scaling laws and compute-optimal models](genai-with-llm-course/course2/4_scaling_laws_compute.md)
   - [5 - Pre-training for domain adaptation](genai-with-llm-course/course2/5_pre-training_model_adaptation.md)
 - course 3 - Fine-tuning LLMs with instruction
   - [1 - Instruction fine-tuning](genai-with-llm-course/course3/1_fine-tuning.md)
   - [2 - Multi-task Instruction fine-tuning](genai-with-llm-course/course3/2_fine-tuning_multi-task.md)
   - [3 - Model Evaluation and Benchmarks](genai-with-llm-course/course3/3_model_evaluation.md)
 - course 4 - Parameter efficient fine-tuning (PEFT)
   - [1 - Parameter efficient fine-tuning (PEFT)](genai-with-llm-course/course4/1_parameter_efficient_fine-tuning.md)
   - [2 - PEFT techniques 1: LoRA](genai-with-llm-course/course4/2_peft_techniques_LoRA.md)
   - [3 - PEFT techniques 2: Soft prommpts](genai-with-llm-course/course4/3_peft_techniques_soft_prompts.md)
   - [Lab 2 - Fine-tuning using PEFT with LoRA](genai-with-llm-course/course4/lab_2_peft_LoRA.md)
 - course 5 - Reinforcement learning from human feedback
   - [1 - Reinforcement learning from hman feedback (RLHF)](genai-with-llm-course/course5/1_reinforcement_learning_from_human_feedback.md)
   - [2 - RLHF: Obtaining feedback from humans](genai-with-llm-course/course5/2_rlhf_obtaining_feedback_from_humans.md)
   - [3 - RLHF: Reward hacking and KL Divergence](genai-with-llm-course/course5/3_rlhf_reward_hacking.md)
   - [4 - Scaling human feedback](genai-with-llm-course/course5/4_scaling_human_feedback.md)
   - [Lab - Fine-Tune FLAN-T5 to Generate More-Positive Summaries](genai-with-llm-course/course5/lab_fine-tune_FLAN-T5_Sumarization.md)
 - course 6 - LLM-powered applications
   - [1 - Model optimizations for deployment](genai-with-llm-course/course6/1_model_optimizations_deployment.md)
   - [2 - Generative AI Project Lifecycle and Level of Effort Cheat Sheet](genai-with-llm-course/course6/2_genai_project_lifecycle_cheat_sheet.md)

  

 - Quiz
   - [1 - LLM Transformers and pre-training scaling laws](genai-with-llm-course/quiz/quiz1.md)
   - [2 - Fine Tuning LLMs with Instruction and PEFT](genai-with-llm-course/quiz/quiz2.md)


 - Slides from DeepLearning.ai (lecture notes)
  - [Week 1 slides](slides/W1.pdf)
  - [Week 2 slides](slides/W2.pdf)
  - [Week 3 slides](slides/W3.pdf)



### Notebooks and code snippet

### Credits and Resources
#### Courses
 - [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) Course from AWS and Deeplearning.ai
 - [Machine Learning specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
 - [Deep Learning specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)

#### Papers
 - [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Google and Toronto University papaer introducing the Transformer architecture, with the core "self-attention" mechanism. This article was the foundation for LLMs.
 - [Transfomer: A Novel Neural Network Architecture for Language understanding](https://blog.research.google/2017/08/transformer-novel-neural-network.html) - Google Blog
 - [BLOOM: BigScience 176B Model](https://arxiv.org/abs/2211.05100) - BLOOM is a open-source LLM with 176B parameters trained in an open and transparent way. In this paper, the authors present a detailed discussion of the dataset and process used to train the model. You can also see a high-level overview of the model [here](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) 
 - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - empirical study by researchers at OpenAI exploring the scaling laws for large language models.
 - [BloombergGPT paper](https://arxiv.org/abs/2303.17564) - LLM trained specifically for the finance domain, a good example that tried to follow chinchilla laws.
 - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf) - Article from Meta AI proposing Efficient LLMs (their model with 13B parameters outperform GPT3 with 175B parameters on most benchmarks)
 - [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) - Study from DeepMind to evaluate the optimal model size and number of tokens for training LLMs. Also known as “Chinchilla Paper”.
 - [FLAN (Fine-tuned LAnguage Net)](https://arxiv.org/abs/2210.11416) - an instruction finetuning method. FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model
 - [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf) - Scaling fine-tuning with a focus on task, model size and chain-of-thought data.
 - [Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://blog.research.google/2021/10/introducing-flan-more-generalizable.html) - This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.
 - [HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/lite/latest/) - HELM is a living benchmark to evaluate Language Models more transparently. 
 - [General Language Understanding Evaluation (GLUE) benchmark](https://openreview.net/pdf?id=rJ4km2R5t7) - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.
 - [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf) - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.
 - [PEFT - Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf) - This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.
 - [LoRA Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) - This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.
 - [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) - The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.

#### Other resources
 - [Hugging Face Tasks](https://huggingface.co/tasks)
 - [Hugging Face models hub](https://huggingface.co/models)
 - [DeepLearning.ai lecture notes](https://community.deeplearning.ai/t/genai-with-llms-lecture-notes/361913)


