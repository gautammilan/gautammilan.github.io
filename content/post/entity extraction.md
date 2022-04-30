+++ 
date = 2022-01-03T21:38:54+05:45
title = "Transactions entity extractor"
description = ""
slug = ""
authors = ["Milan Gautam"]
tags = []
categories = ["Deep Learning"]
externalLink = ""
series = ["Artificial Intelligence"]
+++

<!-- ## Introduction -->
As most of today's world has been digitalized, instead of writing transactions in journals people do it on their computers. In many cases, these transactions have important information about the parties involved in it, so business tries to acquire it by using many techniques. In this article, we will look into one such transaction and try to acquire embedded information inside the transaction using the deep learning model.

![Conventional Method](/images/dataset_image_entity_extractor.png#center "Conventional Method")

As we can see the transaction contains the store number information, usually people had to go through all of these transactions to get the store number ID. But in today's era, it’s not feasible where thousands of transaction happens every single second using various digital means.
  
<br>

## Mapping it to machine learning problem

![Conventional Method](/images/model_idea_entity_extractor.png#center)


So basically we need to build a model which will make the transaction as an input and produce its store number. As we are emphasizing each character and determining whether it is a store number or not. Character LSTM is capable of doing this task, so the ideal choice for solving this problem is a char-LSTM model,  so we will be using it.


![Conventional Method](/images/model_entity_extractor.png#center "Working of Bidirectional LSTM")

<br>

We will approach this problem as binary classification where the position of store numbers present inside the transaction is labeled as 1 and all the other characters as 0. During inference, we will select all the characters as store numbers that have the prediction of 1. The position of the store number is not rigid and it depends upon the characters appearing on both sides of it. So it is important to look at the transaction from left to right and right to left and then select the store number. Therefore bidirectional LSTM is capable of doing that and we will be using it in this project. Similarly, the size of data that we will be working on is really small only 100 data points for train and we will use other 100 data points for validation, so we will use different training approaches to get the best out of it.

![Conventional Method](/images/dataset_distribution_entity_extractor.png#center "Dataset")


## Preprocessing 


### 1. 1.	Removing consecutive Whitespaces
 
We are considering whitespace as a special character, if multiple whitespaces appear in the transaction then they get tokenized using consecutive whitespace tokens, here the model will learn the necessity of adding consecutive whitespaces which is not true. So by removing the consecutive whitespaces altogether we elimate these phenomena. For example 

Before removal: Spark     2001

After removal:  Spark 2001

<br>

### 1.2 Removing names

 All the transactions have some kind of name inside them like ANN TAYLOR FACTORY #2202, MCDONALD'S F1682, etc. For some models, these names accelerate the training and for some, it works counterproductive. Depending upon the model we will be removing names from the transaction.

Before removal: Spark 2001

After removal: 2001

<br>

### 1.3 Tokenization amd Padding

Dictionary is created for all the characters(a,A,c,C...), non-characters(@,#,%,...) and digits(1,2,3,4,..). Two special tokens for whitespace character and padding([PAD]) are also included in the dictionary which makes the total size of the vocabulary 87. Finally, all the characters of the transactions are tokenized using the dictionary.

 We have considered that every transaction will have 50 characters, to make sure every transaction has the same length. If the transaction is smaller than this it is padded using padding character [PAD] and if it is larger than 50 characters then it is truncated.

 <br>

### 1.4 Label Creating

A list of labels is created for each transaction. If the character is a store number character then it is labeled as 1 elsewhere it is labeled as 0. For transactions such as DOLRTREE 2257 00022574, where the characters forming the store number are located at two places. In such a case the store number characters which are located on the leftmost side are labeled as store number characters and the other one is ignored.

Transaction: DOLRTREE 2257 00022574 

label :00000000 1111 00000000

Note: For this example whitespace characters has not be labeled.

<br>

# Models


## 1.	Regular expression model

By addressing different scenario that occur on training dataset different regular expression operation has been performed on the data. The accuracy for this model is:

a.	Training dataset= 97%

b.	Validation dataset= 87%


Just by using regular expression the accuracy of validation data is also really good which indicates there is not much difference between the training and validation set. As the possibility of overfitting is really small, we can even train the model for a large number of epochs.

<br>

## 2.	LSTM Model

There are two inputs to the model one is the token and the second is the attention mask which is a Boolean matrix indicating which tokens have been padded, it is useful when training and calculating loss so that the loss of the padded token doesn’t get included on our overall loss. Similarly, as the store-number character is dependent on the characters on both sides we will use bidirectional LSTM. Bidirectional LSTM produces two outputs for every single cell, one when viewing the characters from the left and the other from the right, we will merge these outputs by taking an average. 


{{< highlight go >}} 
class Bi_LSTM(keras.layers.Layer):
    def __init__(self,units):
        super(Bi_LSTM, self).__init__()
        #Return_sequences: return output of every single cell
        #return_state: return hidden state of every single cell
        
        self.lstm = layers.LSTM(units,return_sequences=True)
        self.bi_lstm= layers.Bidirectional(self.lstm,merge_mode='ave')

    def call(self, inputs,attention_mask):
        #It is important to sending the masking vector in order to indicate which tokens are masked tokens
        output= self.bi_lstm(inputs,mask= attention_mask)
        return output

def create():
    inputs= tf.keras.Input(shape= (preprocessor_train.pad_len),dtype= tf.float32)
    att_mask= tf.keras.Input(shape= (preprocessor_train.pad_len),dtype= tf.bool)
    x= layers.Embedding(len(dictionary),50)(inputs)

    #Here we will be using two LSTM layers each followed by an dropout layer

    x= Bi_LSTM(126)(x, attention_mask= att_mask)
    x= layers.Dropout(0.3)(x)

    x= Bi_LSTM(65)(x, attention_mask= att_mask)
    x= layers.Dropout(0.3)(x)

    x= layers.TimeDistributed(layers.Dense(32))(x,mask= att_mask)
    x= layers.Dropout(0.3)(x)

    output= layers.TimeDistributed(layers.Dense(1,activation='sigmoid'))(x,mask= att_mask)
    model= tf.keras.Model(inputs= [inputs,att_mask],outputs= output)
    return model

model= create()
model.summary()



 {{< /highlight >}}

<br>

![Conventional Method](/images/architecture_entity_extractor.png#center "Architecture")


In this architecture there are two bidirectional LSTM layers each followed by a dropout layer with a probability of 0.3, the output of the second layer is provided as input for the dense layer which is timely distributed. 

<br>

## Evaluation on different loss function 

#### 1.1 Simple BCE loss


Binary cross-entropy loss is the average logistic loss on every prediction. Lets look at the output generated by this loss on validation dataset:

![Conventional Method](/images/simple_BCE_entity_extractor.png#center "Output")

Here the input is labeled as 1 when the output prediction is more than 0.5. Now lets look at the accuracy of train and test set on different value of thresholds:


![Conventional Method](/images/simple_BCE_plot_entity_extractor.png#center "Accuracy vs Threshold")

At threshold 0.6 the train and test accuracy is 80% and 60% respectively.

Assume you have a "Atalanta 23" transaction with a store number of 23. It has 11 characters, including whitespace; we're computing the value of loss for each of these characters, and we're assuming the average loss for this transaction is 0.5. Because there are more non-store number characters than store number characters here, the average loss value shifts toward non-store number characters. As a result, when the model performs incorrect classification on store number characters, its contribution to the average loss remains small, resulting in a small penalization of models, so the model does not focus on predicting the store number character, but only on predicting non-store-number characters.

<br>

#### 1.2 Weighted BCE loss

We will generate a weight value for both labels 0 and 1 so that more weights are assigned to the less often label, and these weights are multiplied to the loss value of that label, to remove the biases of average loss value toward the non-store number character. As a result, assigning more weight to the less common label 1 helps to ensure that both labels receive equal attention. The output it produced at threshold 0.5, as well as accuracies at other thresholds:

![Conventional Method](/images/WBCE_output_entity_extractor.png#center "Output")

<br>

![Conventional Method](/images/WBCE_output_plot_entity_extractor.png#center "Accuracy vs Threshold")

The accuracy is poorer than compared to simple BCE loss, it has the maximum test accuracy of 35% at threshold 0.6.

<br>

#### 1.3 Simple BCE loss after removing the names

Taking the names out of the transaction is another technique to eliminate the imbalance. Almost the majority of the names in the transaction are not store number characters, raising the number of negative labels. We can achieve some equilibrium between label non-store number characters and store number characters by deleting names.

![Conventional Method](/images/RE_output_entity_extractor.png#center "Output")

<br>

![Conventional Method](/images/RE_plot_entity_extractor.png#center "Accuracy vs Threshold")

It worked better than weighted BCE loss producing a test accuracy of 55% at a threshold of 0.6. But still, simple BCE loss outperforms it by 5% accuracy on test data.


<br>

##### 1.4 Dice Loss

![(Conventional Method)](/images/DL_work_entity_extractor.png#center "Dice Loss Working")

We were attempting to calculate a loss value for each character in the logistic loss, however, the characters will be unaware of their relationships. Let's say there are four store number characters in a transaction, three of which are labeled 1, and one is labeled 0. Although the logistic loss function performed admirably in this case, the actual output is inaccurate. Because the logistic loss function is unaware of the link between the characters' loss values, it will simply attempt to minimize the average loss, which is contradictory to the goal. At the same time, it just considers character level loss and ignores the expected store number as a whole. The best loss function for this case appears to be Dice loss which increases the overlap between the predicted sequence of output to the actual output.

![Conventional Method](/images/dice_loss_WNR_output_entity_extractor.png#center "Output")

As we can see the model is predicting unwanted sequence of characters, due to which it's accuracy has been reduced drastically. The test accuracy is only 30%.

![Conventional Method](/images/DL_WNR_output_plot_entity_extractor.png#center "Accuracy vs Threshold")


<br>

#### 1.5 Dice Loss loss after removing the names

The discrepancy between the transaction and the store number is relatively little when the names are removed from the transaction. Dice loss works by increasing the intersection between the actual and anticipated sequences of store number characters. By removing names from the transaction, the length of the transaction is lowered, which minimizes the number of undesirable sequence combinations that we want our dice loss to avoid. 


![Conventional Method](/images/DL_NR_output_plot_entity_extractor.png#center "Output")



We attained the highest accuracy of 72% on the test dataset by doing so, making it the best strategy for training this dataset.

![Alt Conventional Method](/images/plot_entity_extractor.png#center "Accuracy vs Threshold")

<br>

# Traning Parameter
The BCE loss was trained for 1000 epochs with a learning rate of 1e-5. Similarly, with an initial learning rate of 1e-4, the dice loss was trained for 200 epochs. The Adam optimizer was employed, with a polynomial learning rate decay and the first 100 steps of the models operating as warn steps.
