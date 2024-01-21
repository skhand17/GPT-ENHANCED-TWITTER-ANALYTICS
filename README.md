Solution 1: ChatCompletionAPI using “gpt-3.5-turbo”.
To start executing this script, one needs to do the following things:
1. Set up a Virtual environment as ".env" and update your OPENAI_API_KEY variable
with your key value.
2. Install openai, pandas, numpy, and python_dotenv using the following commands:
•
`pip install openai`
•
`pip install pandas`
•
`pip install numpy`
•
`pip install python_dotevn`
3. The libraries required to execute the script are imported.
4. Two pandas data frames are used to store the "airline_train.csv" file and the other to keep
the "airline_test.csv" file.
5. There is a function called "get_airline_name" which utilizes a set of prompts for different
roles, including "system," "user," and "assistant."
6. The Prompt ensures that the ChatCompletion API provides a complete and accurate name
for each tweet. The model utilized here is "gpt-3.5-turbo."
7. A new CSV file called "train_result_airline.csv" is dumped into the system containing the
resulting airline text captured from the above API.
8. The above model is used on test data to see if the correct set of airline names is provided.
9. This approach resulted in around 96% accuracy when the already present "airlines"
column was compared against our generated "ResultAirline" column.
10. The accuracy is determined by matching strings in given column “airlines” with the
generated column “ResultAirline.”
11. This model could be improved by implementing a caching mechanism as tuple with
{tweet, model} as this save our costs of getting airline name detection from the prompt
by connecting with endpoint of Open AI.
12. The script name is solution_one.py and it could be executed using the following
command: `python solution_one.py`.
Solution 2: Fine Tuning existing GPT Models
Script Execution:
Prepare the Training data:
1.Install OpenAI, pandas using following command:
•
`
pip install OpenAI`
.
•
`
pip install pandas`
.
2. Import the required libraries to execute to script.
3.Prepare the prompt format for openAI formatting the prompt as:
{“prompt”: tweet,” completion”: airlines}.
4.Import the prompt format for the entire training dataset.
5. Run below command from cli:
• Install the latest openai by running following command: pip install --upgrade openai.
• Set Your Open AI key: SET OPENAI_API_KEY="<OPENAI_API_KEY>".
• Create fine Tune Model by running following command:
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m
<BASE_MODEL>
• Run the Model by:
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
6. Once completed, Modified model will be generated.
7. Running the test data using the Fine tune Model:
• openai_custom_model function is called for the test data which generates the airline
name in a new column.
8. The accuracy computed using fined tuning on da-vinvi-003 model is around 73%.
