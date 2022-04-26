+++ 
date = 2022-02-02T21:38:54+05:45
title = "Stock Price forecasting On Nabil Bank"
description = ""
slug = ""
authors = ["Milan Gautam"]
tags = []
categories = ["Deep Learning"]
externalLink = ""
series = ["Artificial Intelligence"]
+++

<!-- ## Introduction -->


Nabil bank is a bank located in Nepal, it's been trading in NEPSE(Nepal Stock Exchange ) for the past 20 years. We can easily get this data by going to the NEPSE website. This data contains four features such as the price of the stock when the market opens on a particular date, the maximum value of the stock, the minimum value, and the value of the stock at which the market close on that day.

# Description
There are two types of models that are widely used for time series analysis they are:

1. Single Step model= Here model will look one step into the future. For example, given all the past one month of the stock data model will predict what will be the stock value tomorrow. The dense model and LSTM model are used to evaluate the single-step model.

2. Multi-step model= In a multi-step prediction, the model needs to learn to predict a range of future values. Thus, unlike a single-step model, where only a single future point is predicted, a multi-step model predicts a sequence of the future values. The autoregressive model is used for this task.

Therefore in this project, we analyzed different models and evaluate which works best for our data.

[Code](https://github.com/gautammilan/Stock-price-predictoin-Nabil-Bank-)

[Article](https://gautammilan.github.io/post/stock-price-analysis/)

