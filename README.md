# Neural Network Charity Analysis: Module 19 Challenge

## Overview of Project

### Background and Purpose

Beks is a data scientist and programmer for the non-profit foundation, Alphabet Soup. They're a philanthropic foundation dedicated to helping organizations that protect the environment, improve peoples' well-being, and unify the world. Alphabet Soup has raised and donated over 10 billion dollars in the past 20 years. This money has been used to invest in the life-saving technologies and organize reforestation groups around the world. Beks is in charge of data collection and analysis for the entire organization. Her job is to analyze the impact of each donation and vet potential recipients. This helps ensure that the foundation's money is being used effectively. Unfortunately, not every donation the company makes is impactful. In some cases, an organization will take the money and disappear. As a result, Alphabet's Soup's president, Andy Glad, has asked Beks to predict which organizations are worth donating to, and which are too high risk. He wants her to create a mathematical, data-driven solution that can do this accurately. Beks has decided that this problem is too complex for the statistical and machine learning models she has used. Instead, she will design and train a deep learning neural network. This model will evaluate all types of input data and produce a clear, decision-making result. Beks needs help learning about neural networks and how to design and train these models using the Python TensorFlow library, relying on our past experience with statistics and machine learning to help test and optimize the models. We will be creating a robust, deep learning neural network capable of interpreting large, complex data sets, which will help Beks and Alphabet Soup determine which organizations should receive donations.

We’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

+ EIN and NAME—Identification columns
+ APPLICATION_TYPE—Alphabet Soup application type
+ AFFILIATION—Affiliated sector of industry
+ CLASSIFICATION—Government organization classification
+ USE_CASE—Use case for funding
+ ORGANIZATION—Organization type
+ STATUS—Active status
+ INCOME_AMT—Income classification
+ SPECIAL_CONSIDERATIONS—Special consideration for application
+ ASK_AMT—Funding amount requested
+ IS_SUCCESSFUL—Was the money used effectively

### Resources

- Data Sources: charity_data.csv
- Software: Anaconda 4.13, Jupyter Notebook, Python 3.7.13, Visual Studio Code 1.68.1

## Results

### Data PreProcessing

+ What variable(s) are considered the target(s) for your model?

    - The target variable for our model is column IS_SUCCESSFUL.

+ What variable(s) are considered to be the features for your model?

     - The features variables for our model are the following columns: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

+ What variable(s) are neither targets nor features, and should be removed from the input data?

     - The variables that are neither targets or features that were removed from the input data are columns EIN and NAME.

### Compiling, Training, and Evaluating the Model

+ How many neurons, layers, and activation functions did you select for your neural network model, and why?

    - For the initial model and for attempt #1, we used two hidden layers, comprised of 80 neurons for the first layer and 30 neurons for the second layer.
    - For attempt #2, we used three hidden layers, comprised of 80 neurons for the first layer, 30 neurons for the second layer, and 10 neurons for the third layer. We added a third layer to try to boost the accuracy of the model.
    - For attempt #3, we used three hidden layers, comprised of 100 neurons for the first layer, 50 neurons for the second layer, and 30 neurons for the third layer. We increased the number of neurons per layer to try to boost the accuracy of the model.
    - For all models, we used the relu activation function for all hidden layers to enable nonlinear relationships and the sigmoid activation function for the output layer to produce a probability output.

+ Were you able to achieve the target model performance?

    - No, we were not able to achieve the target model performance. Unfortunately, all attempts were below the target of 75% accuracy.

+ What steps did you take to try and increase model performance?

    - To try to increase model performance, we tried a few different things:
        1. We took the time to bin the values in the column ASK_AMT because there were 8,747 unique values. After binning, we brought that number down significantly to two unique values.
        2. We increased the number of hidden layers from two layers to three layers.
        3. We increased the number of neurons per layer.

## Summary

### High-Level Summary

Our initial model resulted in an accuracy of 72.48%. See Figures 1 and 2 for the model summary and evaluation.

*Figure 1: Initial Model Summary*

![initial_summary](https://user-images.githubusercontent.com/106830513/197676406-9701cd32-c967-418f-a3be-473141aca4cc.png)

*Figure 2: Initial Model Evaluation*

![initial_eval](https://user-images.githubusercontent.com/106830513/197676393-a4e5180f-a175-4afe-b6e5-405ac3e45acb.png)

Our attempt #1 at optimization resulted in an accuracy of 72.63%. See Figures 3 and 4 for the model summary and evaluation.

*Figure 3: Attempt #1 Model Summary*

![att1_summary](https://user-images.githubusercontent.com/106830513/197676363-27355141-b733-407b-af2b-2f8f9b9c76be.png)

*Figure 4: Attempt #1 Model Evaluation*

![att1_eval](https://user-images.githubusercontent.com/106830513/197676358-beb4bc6a-ccfc-439b-980d-c93cfe7837d8.png)

Our attempt #2 at optimization resulted in an accuracy of 72.16%. See Figures 5 and 6 for the model summary and evaluation.

*Figure 5: Attempt #2 Model Summary*

![att2_summary](https://user-images.githubusercontent.com/106830513/197676373-a857c63f-b7ff-4b1c-b217-4b304553a253.png)

*Figure 6: Attempt #2 Model Evaluation*

![att2_eval](https://user-images.githubusercontent.com/106830513/197676368-afbfb3bf-8de3-4a75-a0e6-d0bed855597a.png)

Our attempt #3 at optimization resulted in an accuracy of 72.47%. See Figures 7 and 8 for the model summary and evaluation.

*Figure 7: Attempt #3 Model Summary*

![att3_summary](https://user-images.githubusercontent.com/106830513/197676390-4d05c09a-876e-4657-aba0-92185949fc6f.png)

*Figure 8: Attempt #3 Model Evaluation*

![att3_eval](https://user-images.githubusercontent.com/106830513/197676384-61b0b907-aab5-42b7-86cf-6c7890883ad8.png)

### Model Recommendation

From our first model to our third attempt, our accuracy remained quite steady - between 72.1% and 72.6%. This tells me that it's possible the model isn't performing well because of the data. Our dataset is quite substantial, consisting of 45 columns and over 34,000 different charities. If we wanted to try again on this dataset, I would recommend spending some time determining which columns are most likely to influence a charity's success rate and drop the ones that are low impact, to help make the dataset smaller. Using a small sample of the dataset, we could try a supervised machine learning model that could help provide the importances of each feature. Or I would recommend trying a random forest model - it handles large datasets well, can easily handle outliers and nonlinear data, has faster performance, and is more efficient to code.
