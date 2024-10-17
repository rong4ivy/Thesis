import dspy
from dspy.teleprompt import BootstrapFewShot
from openai import OpenAI
import pandas as pd 
import json
import pandas as pd
from tqdm import tqdm
import re
import random
import ast
 

turbo = dspy.OpenAI(
    model="deepseek-chat",api_base="https://api.deepseek.com", 
    api_key=" ",
    stop='\n\n',model_type='chat',
           temperature=0.2,
            max_tokens=300,
            top_p=0.5,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stream = False,
        )

dspy.settings.configure(lm=turbo)


# Step 1: Define the signatures for each module
#
class BasicQA(dspy.Signature): 
    """ To assist question answering, please create a 2D layout diagram of the described spatial scene. 
  This diagram should visually represent the placement and relative positions to each other, use the symbols for relationship,
  up: ↑, down: ↓, left: ←, right: →, Upper-left: ↖, Upper-right: ↗,lower-left: ↙,lower-right: ↘ 
    """
    context = dspy.InputField(desc="descriptions of 8 potential spatial relations between 2 entities. Possible relations are: overlap, above, below, left, right, upper-left, upper-right, lower-left, and lower-right. If a sentence in the story is describing clock-wise information, then 12 denotes above, 1 and 2 denote upper-right, 3 denotes right, 4 and 5 denote lower-right, 6 denotes below, 7 and 8 denote lower-left, 9 denote left, 10 and 11 denote upper-left. If the sentence is describing cardinal directions, then north denotes above, east denotes right, south denotes below, and west denotes left.")
    question = dspy.InputField(desc="the question to be answered")
    #choices = dspy.InputField(desc="the candidate answers to choose")
    answer = dspy.OutputField(desc="1 word")
    
class Vision(dspy.Module):  # let's define a new module
    def __init__(self):
        super().__init__()
        # here we declare the chain of thought sub-module, so we can later compile it (e.g., teach it a prompt)
        self.generate_answer = dspy.ChainOfThought(BasicQA)
        
    def forward(self, context, question):
        
        return self.generate_answer(context= context, question = question)

        #answer = self.generate_answer(context= context, choices =choices).answer
        #return dspy.Prediction(answer=answer)

# Step 3: Prepare the dataset

with open('clean/qa4_test.json', 'r') as file:
    data = json.load(file)

# Transform data into a list of records
clean_data = [value for key, value in data.items()]

# Create examples from the clean data
examples = [dspy.Example(
    {"context": " ".join(r["story"]), "question": r["question"], "answer": r["label"]}
    ).with_inputs("context", "question") for r in clean_data]
print(f"There are {len(examples)} examples.")
train = examples[0:10]
val = random.sample(examples[10:], 50)


# Step 4: Define the metric for evaluation

from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

# Step 4: Define the metric for evaluation
def validate_answer(example, pred, trace=None):
    
    pred_cleaned = re.sub(r"^(Answer:\s*)", "", pred.answer, flags=re.IGNORECASE).strip()
    # Perform fuzzy matching, converting both strings to lowercase
    similarity_score = fuzz.ratio(example.answer.lower(), pred_cleaned.lower())
    # Return True if the similarity score is above the threshold, False otherwise
    return similarity_score >= 80

# Step 5: Use a teleprompter for optimization

teleprompter = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=2)

cot_compiled = teleprompter.compile(Vision(), trainset=train)

# Step 6: Prepare the evaluation set
devset = val  # Assuming you use the same examples for evaluation

# Step 7: Create an evaluator
evaluator = Evaluate(
    devset=val,
    metric= validate_answer,
    #num_threads=32,
    display_progress=True,
    display_table=10
)

# Step 9: Evaluate the optimized pipeline
evaluation_results = evaluator(cot_compiled)
print("optimized",evaluation_results)
evaluation = evaluator(Vision())
print("Zero",evaluation)