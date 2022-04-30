+++ 
date = 2022-01-03T21:38:54+05:45
title = "Transactions entity extractor"
description = ""
slug = ""
authors = ["Milan Gautam"]
tags = ["Deep Learning"]
categories = ["Deep Learning"]
externalLink = ""
series = ["Artificial Intelligence"]
+++

<!-- ## Introduction -->
Because most of today's world has been digitalized, instead of recording transactions in journals, people record them on their computers. In many circumstances, these transactions contain critical information about the parties involved, thus businesses attempt to obtain it through a variety of means. In this study, we will examine one such transaction and attempt to acquire embedded information within it using a deep learning model.

![Conventional Method](/images/dataset_image_entity_extractor.png#center "Conventional Method")

So, in this study, I created a model that takes a transaction as an input and generates its store number. While highlighting each character and evaluating whether or not it is a store number. Because character LSTM is capable of doing this task, a char-LSTM model is the best solution for handling this problem, and we will use it.

[Github Code](https://github.com/gautammilan/Transactions-entity-extractor)

The following article details the steps I took to address this issue:

[Article](https://gautammilan.github.io/post/entity-extraction/)
