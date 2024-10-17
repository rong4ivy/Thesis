
# Load the JSON data from a file
with open('SpartQA_human/human_test.json', 'r') as file:
    data = json.load(file)

# Prepare the list of records for the DataFrame
records = []
for item in data['data']:
    context = " ".join(item['story'])  # Join the list into a single string
    for question in item['questions']:
        choices = question['candidate_answers']
        if not choices:  # If choices is an empty list or None
            choices = ['Yes', 'No', 'DK']
        ground_truth = question.get('answer', None)
        answer = [choices[i] for i in ground_truth] if isinstance(ground_truth, list) and all(isinstance(i, int) for i in ground_truth) else ground_truth
        transformed_item = {
            'story': context,
            'question': question.get('question', None),
            'q_type': question.get('q_type', None),
            'reasoning_type': question.get('reasoning_type', None),
            'choices': choices,
            'answer': answer
        }
        records.append(transformed_item)
        
df = pd.DataFrame(records)
df.to_csv('SparQA_test.csv', index=False)


# Filter out the rows with null, None, or empty string values in the specified column
#df = df[df['answer'].apply(lambda x: x is not None and (isinstance(ast.literal_eval(x), list) and len(ast.literal_eval(x)) != 0 if isinstance(x, str) else False))]

#df = df[df['answer'].notna() & (df['answer'] != "")]

# Save the filtered DataFrame back to a CSV file

print(len(df))

df2 = pd.read_csv('SparQA_test.csv')
df2 = df2[df2['answer'].apply(lambda x: x is not None and (isinstance(ast.literal_eval(x), list) and len(ast.literal_eval(x)) != 0 if isinstance(x, str) else False))]
print(len(df2))
df2.to_csv('SparQA_test.csv')
# Convert the DataFrame to a dictionary
clean_data = df2.to_dict(orient='records')