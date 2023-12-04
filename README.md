# Gen AI LLM

This repo inclueds courses, code snippets from various sources to better learn and build an intuition about Large Language Models
I first enrolled in a course from Deeplearning.ai called [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) Course from AWS and Deeplearning.ai.
The main reason is my prior experience with Deeplearnin.ai as I gained a lot of knowledge from 2 prior courses from Andrew Ng and his team, namely: 
 - [Machine Learning specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
 - [Deep Learning specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/)

### Course 1 - Generative AI with Large Language Model
Below are some key notes from [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms)

##### Intro
Generative AI and LLMs specifically are a general purpose technology. That means that similar to other general purpose technologies like deep learning and electricity, is useful not just for a single application, but for a lot of different applications that span many corners of the economy. Similar to the rise of deep learning that started maybe 15 years ago or so, there's a lot of important where it lies ahead of us that needs to be done over many years by many people, to identify use cases and build specific applications.

#### Transformer Network
The [Transfomer: A Novel Neural Network Architecture for Language understanding](https://blog.research.google/2017/08/transformer-novel-neural-network.html)  blog and corresponding paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) from Google were released on aug 2017.
This is the foundation of LLM and Foundational Models. And it's amazing how long the transformer architecture has been around and it's still state of the art for many models

From the Blog: "Neural networks, in particular recurrent neural networks (RNNs), are now at the core of the leading approaches to language understanding tasks such as language modeling, machine translation and question answering. In “Attention Is All You Need”, we introduce the Transformer, a novel neural network architecture based on a self-attention mechanism that we believe to be particularly well suited for language understanding.
In our paper, we show that the Transformer outperforms both recurrent and convolutional models on academic English to German and English to French translation benchmarks. On top of higher translation quality, the Transformer requires less computation to train and is a much better fit for modern machine learning hardware, speeding up training by up to an order of magnitude."

We will learn the intuitions behind some of these terms you may have heard before, like multi-headed attention. What is that and why does it make sense? And why did the transformer architecture really take off. To note transformer architecture also helped on other modalities than text/NLP like vision.
To also note we can also use a transformer architecture with smaller foundational models than the large ones that have 100 of billion of parameters for single tasks like summarilizing dialog.

### Notebooks and code snippet

### Credits and Resources
#### Courses
 - [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) Course from AWS and Deeplearning.ai

#### Papers
 - [Attention is All You Need] (https://arxiv.org/abs/1706.03762) Google paper
 - [Transfomer: A Novel Neural Network Architecture for Language understanding](https://blog.research.google/2017/08/transformer-novel-neural-network.html) Google Blog
