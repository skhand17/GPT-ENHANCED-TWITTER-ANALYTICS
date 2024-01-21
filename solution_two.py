#!/usr/bin/env python
# coding: utf-8

# In[78]:


import openai
import os
import pandas as pd
import ast
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')


# In[92]:


data = pd.read_csv('airline_train.csv')


# In[93]:


question,answer = data['tweet'],data['airlines']


# In[96]:


#Preparing the Prompt format
openai_format = [ {"prompt":q,"completion":a} for q,a in zip(question,answer)]


# In[11]:


response = openai.Completion.create(

            model = 'text-davinci-003',
            prompt = qa_openai_format[1]['prompt'],
            max_tokens = 250,
            temperature = 0)


# In[97]:


#importing the Prompt as per above prompt format for the training data set:
import json
with open('example_traning_data_airline.json','w') as f:
    for entry in openai_format[:]:
        f.write(json.dumps(entry))
        f.write('\n')
    
    


# In[19]:


fine_tunes_model = "davinci:ft-personal-2023-06-23-01-57-18"


# In[48]:


def openai_custom_model(text):

    response = openai.Completion.create(

                model = fine_tunes_model,
                prompt = f' {text} . Please note not to include the Duplicate Result in the response',
                temperature = 0)
    return response['choices'][0]['text']


# In[49]:


test_data = pd.read_csv('airline_test.csv')


# In[51]:


test_data['result'] = test_data['tweet'].apply(openai_custom_model)


# In[56]:


test_data.to_csv('test_result.csv',index = False)


# In[84]:


def convert(str):
  list_of_words = str.replace('[','').replace(']','').replace('\'','').split(",")
  for index, word in enumerate(list_of_words):
    list_of_words[index] = word.strip()
  return list_of_words


# In[ ]:





# In[85]:


def remove_duplicate(str):
  new_list = []
  list_of_words = str.replace('][',',').replace('[','').replace(']','').replace('\'','').split(",")
  for word in list_of_words:
    if word not in new_list:
      new_list.append(word.strip())
  return new_list
        
    


# In[90]:


def multi_label_accuracy(y_true, y_pred):
  """Calculates the accuracy of a multi-label classifier."""
  correct = 0
  for pred in y_pred:
    if pred in y_true:
        correct = correct + 1
  if len(y_true) == len(y_pred) and correct == len(y_true):
    return 1
  else:
    return 0;


# In[89]:


predicted = test_data['result'].apply(remove_duplicate)
actual = test_data['airlines'].apply(convert)


# In[91]:


total_accuracy = 0
for (truth,prediction) in zip(actual,predicted):
  total_accuracy = total_accuracy + multi_label_accuracy(truth, prediction)
print(total_accuracy/len(actual))


# In[ ]:





# In[ ]:




