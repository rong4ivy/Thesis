import dspy
import pandas as pd
import json
from typing import List, Dict, Any
import groq 
import os

#Initialize the llm
turbo= dspy.GROQ(model="groq/llama3-70b-8192", api_key = "",  
            max_tokens=4096,
            temperature = 0.1,
            top_p=1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stream = False,)
dspy.configure(lm=turbo)

# the following is two stages of pipeline, more stages can be added. 
# It is important to set state dict to store all the inputs and outputs so that every stage can share the memory during the whole pipeline.


class Convert(dspy.Module):
    class Signature(dspy.Signature):
        prompt_1 = dspy.InputField(desc="The prompt for the task")
        context = dspy.InputField(desc="Spatial description of scene and question, parsed it to facts")
        question = dspy.InputField(desc="The question to be parsed into query based on the context")
        facts = dspy.OutputField(desc="Limit your output to 3000 tokens. Please do not include prolog or asp in the beginning of output")

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.predictor = dspy.ChainOfThought(self.Signature)

    def forward(self, prompt_1, context, question):
        result = self.predictor(prompt_1=prompt_1, context=context, question=question)
        facts = result.facts
        self.state['convert'] = {
            'facts': facts,
            'context': context,
            'question': question,
        }
        return dspy.Prediction(facts=facts)

class ASP(dspy.Module):
    class Signature(dspy.Signature):
        facts = dspy.InputField(desc="Initial ASP facts, query")
        prompt_2 = dspy.InputField(desc="The prompt for the task")
        asp= dspy.OutputField(desc="Complete ASP code. Limit your output to 3000 tokens.")

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.predictor = dspy.ChainOfThought(self.Signature)
    def forward(self, facts, prompt_2):
        result = self.predictor(facts=facts, prompt_2=prompt_2)
        asp= result.asp
        self.state['asp'] = {
            'asp': asp,
        }
        return dspy.Prediction(asp=asp)

class Pipeline(dspy.Module):
    def __init__(self, state, max_iters=3):
        super().__init__()
        self.state = state
        self.convert = Convert(state)
        self.asp = ASP(state)
        self.max_iters = max_iters

    def forward(self, context, question, prompt_1, prompt_2):
        # Convert the natural language description into ASP facts and query
        convert_result = self.convert.forward(prompt_1=prompt_1, context=context, question=question)
        facts = convert_result.facts

        for _ in range(self.max_iters):
             # revise the previous results through loops
            revise_result = self.asp.forward(facts=facts, prompt_2=prompt_2)
            asp = revise_result.asp
        return dspy.Prediction(asp=asp, error=None)

def process_examples(examples: List[dspy.Example], pipeline: Pipeline) -> List[Dict[str, Any]]:
    results = []
    for example in examples:
        context = example.get('context')
        question = example.get('question')
        prompt_1 = example.get('prompt_1')
        prompt_2 = example.get('prompt_2')
        prediction = pipeline(context=context, question=question, prompt_1=prompt_1, prompt_2=prompt_2)
        
        result = {
            "context": context,
            "question": question,
            "predicted": prediction.asp,
            "actual_answer": example.get('answer'),
            "error": prediction.error
        }
        results.append(result)
        
        # Save individual ASP code to a JSON file
  
    return results

def main():
    # Prepare the dataset
    df2 = pd.read_csv(' *.csv')
    clean_data = df2.to_dict(orient='records')
    
    examples = [
        dspy.Example(
            prompt_1 = prompt_facts,
            prompt_2 = prompt_rules,
            context=r["Story"],
            question="".join(r["Question"]),
            choices="".join(r["Candidate_Answers"]),
            answer="".join(r["Answer"])
        ).with_inputs("context", "question", "prompt_1", "prompt_2", "choices")
        for r in clean_data 
    ]

    # Initialize and run the pipeline
    state = {}
    pipeline = Pipeline(state)
    
    results = process_examples(examples, pipeline)
    
    with open("complete_ASP.json", "w") as jsonfile:
        json.dump(results, jsonfile, indent=4)
                
                # json.dump({"context": context, "question": question, "asp_code": prediction.asp_code}, jsonfile)
                # jsonfile.write("\n")  # Add a newline for separation

    # Save results
if __name__ == "__main__":
    main()