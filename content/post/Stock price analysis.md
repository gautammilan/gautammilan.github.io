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
In statistical terms, time series forecasting is the process of analyzing the time series data using statistics and modeling to make predictions and informed strategic decisions. It falls under Quantitative Forecasting. Examples of Time Series Forecasting are weather forecast over next week, forecasting the closing price of a stock each day etc. In this article we will see different types of models which can be used for this analysis and pick what is best for what situations.  



## Time series data

Time series data are simply measurements or events that are tracked, monitored, downsampled, and aggregated over time. This could be server metrics, application performance monitoring, network data, sensor data, events, clicks, trades in a market, and many other types of analytics data. We will be taking stock price data to perform our analysis.

Nabil bank is a bank located in Nepal, it's been trading in NEPSE(Nepal Stock Exchange ) for the past 20 years. We can easily get this data by going to the NEPSE website. This data contains four features such as the price of the stock when the market opens on a particular date, the maximum value of the stock, the minimum value, and the value of the stock at which the market close on that day.

![Image of the data](/images/stock/stock_dataframe.png#center)
## Preprocessing 

### 1. Normalization
We need to normalize between 0-1, to remove the problem which arises if the features having in different scales. But when normalizing validate and test data don't use the validate.max() or text.max() and validate.min() or text.min() for their respective normalization use train.max and train.min for both of them. Because we can't look at the validate or test dataset they are unknown to us. The important thing to note here is that the normalization has been done on the input feature only not on the label, the model will predict the actual value of stock.


### 2. Sliding window

To perform Supervised learning the dataset should have inputs and its corresponding label. Data windowing is a popular technique for converting historical data like time series to data suitable for supervised learning. It works as it sounds, we select a window for inputs and feed the model the data which has been selected into that window and the model will try to predict the label for that window.
The main features of the input windows are:

•	The width (number of time steps) of the input and label windows.

•	The time offset between them.

•	Which features are used as inputs, labels, or both.

Depending on the task and type of model we may want to generate a variety of data windows. Here are some examples:
1.	A model that makes a prediction one hour into the future given six days of history,  would need a window like this:

![Example_1](/images/stock/example1.png#center)

2.	Similarly, to make a single prediction 24 days into the future, given 24 days of history, we might define a window like this:

![Example_2](/images/stock/example2.png#center)

[source](https://www.tensorflow.org/tutorials/)


Therefore, depending upon the task and model we can generate varieties of inputs which helps to reduce the redundancy of code as by defining an data window using a class.



## Models
In time series forecasting depending upon the number of steps we are going to do the prediction for the models can be classified into two types:

### 1. Single step Model
In single step model, model will look one step into the future. For example given all the past one month of stock data model will predict what will be the stock value tomorrow. For this task we will be using models like:

#### 1.1	Dense model:
A single dense layer is a single layer of fully connected neural network. Here, we will be sending our Inputs of specific input width into multiple dense layer and finally the output of these dense layer will be send though a single neuron dense layer to produce a single step output. It is an regression problem where we take Open, Close, Low and High as input to predict the closing value of the stock.


{{< highlight go >}} 
def dense_func(input_shape):
  input= tf.keras.Input(shape= tf.constant(input_shape))
  x= tf.keras.layers.Flatten()(input)

  #Basically there are four dense layer each followed by an dropout layer
  x= tf.keras.layers.Dense(units=556, activation='relu')(x)
  x= tf.keras.layers.Dropout(0.2)(x)

  x= tf.keras.layers.Dense(units=228, activation='relu')(x)
  x= tf.keras.layers.Dropout(0.2)(x)

  x= tf.keras.layers.Dense(units=128, activation='relu')(x)
  x= tf.keras.layers.Dropout(0.2)(x)

  x= tf.keras.layers.Dense(units=64, activation='relu')(x)
  x= tf.keras.layers.Dropout(0.2)(x)

  output= tf.keras.layers.Dense(units=1)(x)
  model= tf.keras.Model(inputs= input,outputs= output)  
  return model
 {{< /highlight >}}

![Architecture](/images/stock/desce_code.png#center)
##### a. Hyperparameter tuning
One of the most important hyperparameter for stock price prediction is the number of days that the model sees to make future prediction ie input_width. This hyperparameter value is calculated by training the model on different number of input width and the model which produces lowest loss it is selected.

![Input width vs Mean square Error(MSE)](\images\stock\hyperparameter_for_dense_model.png#center)

We trained the model on input width [3,5,8,15,18,22,25,28,31] and among them the minimum value of MSE was obtained with the 3. So, the input width for the dense model is selected as 3.


##### b. Evaluation 
At input width 3, the label and prediction for Close value of stock look like this:
![label vs prediction](images\stock\Dense_model_ground_truth_vs_prediction.png#center)


#### 1.2 LSTM model
A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data. RNNs process a time series step-by-step, maintaining an internal state from time step to time step.Let’s see understand how RNN will process time series data:

![LSTM modelS](images\stock\RNN.png#center)

Here the RNN/LSTM is trained on every single input step, as a result, it makes the model more robust to changing landscape which is common in the stock dataset. It will take stock prices[Open, Close, High, Low]  as input for the first day and predict the Close value for the second day. Similarly, a second time stamp will take the feature vector generated from the first time stamp and second days inputs to predict the 3rd step value and so on until it predicts one step into the future.

{{< highlight go >}} 
def lstm_model(input_shape):
    
    inp= Input(shape=input_shape) #BATCH,TIMESTAMP,FEATURES
    x= tf.keras.layers.LSTM(128,return_sequences=False,name= 'LSTM')(inp)#batch,timestamp,32
    x= Dense(units=256, activation='relu',name= 'Dense1')(x)
    x=Dense(units=64, activation='relu',name= 'Dense2')(x)

    x=Dense(units=32, activation='relu',name= 'Dense3')(x)
    out= Dense(units=1)(x)
    model= Model(inp,out)
    return model

 {{< /highlight >}}

![Architecture](\images\stock\lstm_code.png#center)

##### a. Evaluation
The minimum value of loss was obtained at input width 11 and its MSE value is similar dense model. Let's look it label and prediction plot:
![label vs prediction](images\stock\rnn_ground_truth_vs_prediction.png#center)


### 2 Multi-step model
In a multi-step prediction, the model needs to learn to predict a range of future values. Thus, unlike a single-step model, where only a single future point is predicted, a multi-step model predicts a sequence of the future values. There are two rough approaches to this:

2.1.	Single-shot Model
 Single-shot Model makes prediction of the entire time series at once. It is a time machine that can jump to any day into the future.

2.2.	Autoregressive predictions where the model only makes single-step predictions and its output is fed back as its input. We can see it as a time machine that can't directly jump to any future date, instead, it had to go through each of the previous dates until it reaches the required future date. 

For example, a person is living in 2012 who wants to go to 2022, if he used its single-shot time machine he can directly go to the year 2022 but as the machine doesn't have any information about the jumped years its prediction events may be different from the actual events. But on the other hand, if he used its autoregressive time machine, the time machine will take him to the year 2013 and then 2014 until he reaches the year 2022, therefore the machine learns information about the intermediate year also which helps to improve the prediction significantly.

![Autoregressive Model](images\stock\autoregressive.png#center)

{{< highlight go >}} 
class denseLayers(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense1= layers.Dense(256)
        self.dense2= layers.Dense(128)
        self.dense3= layers.Dense(32)
        self.dense4= layers.Dense(1)
    
    def call(self,inputs):
        x= self.dense1(inputs)
        x= layers.Dropout(0.1)(x)
        
        x= self.dense2(x)
        x= layers.Dropout(0.1)(x)
        
        x= self.dense3(x)
        x= layers.Dropout(0.1)(x)
        
        x= self.dense4(x)
        return x

def AutoRegressive_func():

  class AutoRegressive(tf.keras.Model):
      def __init__(self, units,output_steps):
          super().__init__()
          self.unit= units
          self.out_step= output_steps
          self.lstm_cell= layers.LSTMCell(self.unit)

          self.lstm_layer= layers.LSTM(128,return_state=True)

          self.dense_layer= denseLayers()

          
      def call(self,inputs,training= True):
          '''
          input= [batch,timestamp,features]
          '''
          predictions= []
          output,state_h,state_c= self.lstm_layer(inputs) 
          #output=[batch,units], similarly output= state_h

          state= [state_h,state_c] # The state
      
          prediction= self.dense_layer(output) 
          predictions.append(prediction)
          
          #Now iterating through the every step
          for i in range(self.out_step-1):
              output,state= self.lstm_cell(output,state,training)
              prediction= self.dense_layer(output) #Prediction= [batch,1] As we are outputting "Close" value at every time stamp
              predictions.append(prediction)
              

          # predictions.shape => (time, batch, features)
          predictions = tf.stack(predictions)
          
          # predictions.shape => (batch, time, features)
          predictions = tf.transpose(predictions, [1, 0, 2])
          return predictions
  return AutoRegressive
 {{< /highlight >}}

In the code, we can see we send the input to an LSTM layer which produces an output and state of the last LSTM cell. These output and state vectors are sent to an LSTM cell to forecast the price for a single day. Similarly, the next LSTM cell executes the previous cell output as input, and the state of the previous cell gets initialized as its initial state. It goes on until we predicted the whole range of output.



#### a. Hyperparamter Tuning
In previous single step model, the minimum value of MSE was obtain when the input width is small but interesting in autoregressive model as the input width increase the MSE reduces. Therefore, the model performs the best when it's looking large number of previous date data. 

![Input width vs Mean square Error(MSE)](images\stock\hyperparameter_for_autoregressiv_model.png#center)

We can see the model performs the best when the input width is 31.

#### b. Evaluation
![Plotting Close value for consecutive days](images\stock\autoregressive_prediction.png#center)

The model is taking a consecutive input of the past 31 days and it is predicting the next 3 days. The difference between the actual price and the predicted price is not much. Therefore depending upon the task we can select an appropriate model and do the task.

