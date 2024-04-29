# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 3000,
    "return_full_text": False,
}

# %%
from datasets import load_dataset
import pandas as pd

math_subjects = [
    'abstract_algebra',
    'college_mathematics',
    'econometrics',
    'elementary_mathematics',
    'formal_logic',
    'high_school_mathematics',
    'high_school_statistics'
]
df_total = pd.DataFrame()
for subject in math_subjects:
    dataset = load_dataset("cais/mmlu", subject)
    for split in dataset.keys():
        df = dataset['test'].to_pandas()
        df["split"] = split
        df_total = pd.concat([df_total, df])

# Save the dataset
filename = "mmlu_math.csv"
df_total.to_csv(filename, index=False)

# %%
import pandas as pd
from tqdm import tqdm

# Load the dataset
filename = "mmlu_math.csv"

df = pd.read_csv(filename)
if 'rationale' not in df.columns:
    df['rationale'] = pd.NA

# Loop through test split
df_dev = df[df["split"] == "dev"]

for i, row in tqdm(df_dev.iterrows()):
    if pd.notna(row["rationale"]):
        continue

    messages = [
        {"role": "user", "content": row["question"]},
    ]

    output = pipe(messages, **generation_args)
    rationale = output[0]['generated_text']
    
    # Update the DataFrame immediately
    df.at[i, "rationale"] = rationale
    df.to_csv(filename, index=False)
    
# %%
import matplotlib.pyplot as plt

# Create a histogram from character lengths of the rationale column
rationale_lengths = df['rationale'].str.len()
plt.hist(rationale_lengths, bins=20)
plt.xlabel('Character Length')
plt.ylabel('Frequency')
plt.title('Histogram of Rationale Lengths')
plt.show()

# %%
filename = "mmlu_math copy 2.csv"

df = pd.read_csv(filename)
if 'guess' not in df.columns:
    df['guess'] = pd.NA

# Loop through test split
df_dev = df[df["split"] == "test"]