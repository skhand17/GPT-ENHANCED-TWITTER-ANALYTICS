#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing all the necessary Libraries
import openai
import json
from dotenv import dotenv_values
config = dotenv_values(".env")
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
import json
import tiktoken
import pandas as pd
import numpy as np
import json
import re


# In[2]:


### Configuring the openai api key by fetching it from the virtual environment
openai.api_key = config['openai_api_key']


# In[3]:


### Reading and the train and test data sets
train_data_frame = pd.read_csv("airline_train.csv")
test_data_frame = pd.read_csv("airline_test.csv")


# In[4]:


# The function is decorated with the @retry decorator, which retries the function in case of an error, using an exponential backoff strategy.
# It will make up to 6 attempts with random exponential waiting times between 1 and 20 seconds.

# To use this function, you can call it with a tweet as the prompt parameter, and it will return the extracted airline name(s) in a JSON response format. 
# Make sure you have the necessary dependencies installed and have access to the OpenAI API.

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_airline_name(prompt):
    
    # This improves the performance
    prompt = prompt.replace("\n", " ")
    
    example = """
    [
        {"airline": "American Airlines"}
    ]
    """
    
    example_2 = """
    [
        {"airline" : "US Airways"}
    ]
    """
    
    example_3 = """
    [
        {"airline" : "JetBlue Airways"}
    ]
    """
    
    example_4 = """
    [
        {"airline" : "US Airways"}
    ]
    """
    example_5 = """
    [
        {"airline" : "United Airlines"},
        {"airline" : "Delta Air Lines"}
    ]
    """
    
    
    messages = [
        {"role" : "system" , "content" : """You are a helpful assistant in extracting airline name from a tweet.
        You should extract the airline name(s) from a given text. You should only return a JSON response, where
        each element follows this format:{"airline": [<airline_name>]}.
        If answer is United then give it as United Airlines.\
        If answer is US Air then give it as US Airways.\
        if answer is SouthwestAir then give it as Southwest Airlines.\
        if answer is American Air then give  it as American Airlines.\
        if answer is VirginAmerica then give it as Virgin America.\
        if answer is Jet Blue then give it as JetBlue Airways.\
        if answer is Delta then give it as Delta Air Lines.\
        Please note not to include duplicate airline names in a single tweet.
        """},
        {"role" : "user", "content" : "@AmericanAir we have 8 ppl so we need 2 know how many seats are on the next flight. Plz put us on standby for 4 people on the next flight?"},
        {"role" : "assistant", "content" : example},
        {"role" : "user" , "content" : "US Air @AskPayPal When will it be marked completed. I\'m scared I\'m going to lose my reservation or my money is going to be refunded"},
        {"role" : "assistant" , "content" : example_2},
        {"role" : "user" , "content" : "Jet Blue I don't know- no one would tell me where they were coming from - I would guess so as that's where we had all the changes to flights"},
        {"role" : "assistant" ,"content" : example_3},
        {"role" : "user", "content" : "USAirways   My email gets me a canned response that it takes 10 days for mileage to get credited --- It's been three months and my miles"},
        {"role" : "assistant" ,"content" : example_4},
        {"role" : "user" ,"content" : "United Airlines considering it. Currently gold on Delta. Why should I make the jump for an upcoming flight from SFO to Singapore?"},
        {"role" : "assistant", "content" : example_5},
        {"role": "user", "content" : f"Extract the name of the airline(s) in the following tweet : {prompt}"}
    ]

    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages,
                temperature=0.0
                )
    airlinelist = response["choices"][0]["message"]["content"]
    return airlinelist


# In[5]:


### The response from the above function is dumped into a new dataframe column called as "ResultAirline"
train_data_frame["ResultAirline"] = train_data_frame["tweet"].apply(get_airline_name)


# In[6]:


def regex_airline_name(d1):
    """
    Extracts the content within curly braces {} in a given string.

    Args:
        d1 (str): The input string from which to extract the content within curly braces.

    Returns:
        list: A list containing the extracted content within curly braces.

    Example:
        >>> regex_airline_name("This is a {sample} string with {multiple} occurrences.")
        ['sample', 'multiple']
    """
    d2 = re.findall(r'\{([^}]*)\}',d1)
    return d2
    


# In[7]:


train_data_frame["ResultAirline"] = train_data_frame["ResultAirline"].apply(regex_airline_name)


# In[8]:


train_data_frame.to_csv("train_result_airline.csv", index=False)


# In[9]:


### Applying the model developed on Training solution above to generate tweets on the test data set
test_data_frame["ResultAirline"] = test_data_frame["tweet"].apply(get_airline_name)


# In[10]:


test_data_frame["ResultAirline"] = test_data_frame["ResultAirline"].apply(regex_airline_name)
test_data_frame.to_csv("test_result_airline.csv", index=False)


# In[11]:


### Applied String manipulation logic to just fetch the name of the airlines
def get_airline_names(response_list):
    names = []
    for response in response_list:
        splitted_comma_strs = response.replace('\'','').split(',')
        for splitted_comma_str in splitted_comma_strs:
            splitted_str = splitted_comma_str.split(':')
            name = splitted_str[1].replace(']','')
            name = name.replace("\"","")
            name = name.strip()
            names.append(name)
    return names


# In[12]:


test_data_frame["ResultAirline"] = test_data_frame["ResultAirline"].apply(get_airline_names)


# In[13]:


def multi_label_accuracy(y_true, y_pred):
    """
    Please note that the function assumes the labels (y_true and y_pred) are provided as lists. 
    It compares the predicted labels (y_pred) against the true labels (y_true) and calculates the accuracy score as a float.
    The function returns a value of 1 if all labels are predicted correctly and have the same length, and 0 otherwise.
    """
    correct = 0
    for pred in y_pred:
        if pred in y_true:
            correct = correct + 1
    if len(y_true) == len(y_pred) and correct == len(y_true):
        return 1
    else:
        return 0;


# In[14]:


def convert_to_list(str):
    list_of_words = str.replace('[','').replace(']','').replace('\'','').split(",")
    for index, word in enumerate(list_of_words):
        list_of_words[index] = word.strip()
    return list_of_words


# In[15]:


actual = test_data_frame['airlines'].apply(convert_to_list)


# In[16]:


predicted = test_data_frame["ResultAirline"]


# In[17]:


total_accuracy = 0
for (truth,prediction) in zip(actual,predicted):
  total_accuracy = total_accuracy + multi_label_accuracy(truth, prediction)
print(f"The total accuracy on test data set : {total_accuracy/len(actual)}")


# In[ ]:




