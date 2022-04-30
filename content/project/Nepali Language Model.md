+++ 
date = 2022-04-02T21:38:54+05:45
title = "Nepali Langauge Model"
description = ""
slug = ""
authors = ["Milan Gautam"]
tags = ["Deep Learning"]
categories = ["Deep Learning"]
externalLink = ""
series = ["Artificial Intelligence"]
+++

With the introduction of BERT in 2018, the machine became as capable of understanding natural language as an ordinary human being, making it feasible to sift through massive corpora of text data and extract meaningful information from it. Due to high data requirements and computational resources, only a few languages have made this transition into the modern era, while many languages, such as Nepali, are still in the prehistoric days of NLP. 

After learning about it, it became my goal to design a Nepali language model and make it open source so that anyone could use it. I began working on the dream by gathering 14.5 GB of Nepali Corpus from over 50 Nepali news websites. To assess the performance of various input length language models, we trained two models, one with 128 token lengths and the other with 512 token lengths, using cloud TPUs. To test our model, we are now developing a GLUE-like evaluation task for the Nepali language. Because this is an active project, I may not be able to release the code.

