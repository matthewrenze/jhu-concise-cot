# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the parameters
model_type = "gpt-3.5"
prompt_type = "Composite"
exam_type = None
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter the model
if model_type is not None:
    details = details[details["Model"] == model_type]

# Filter the prompt
if prompt_type is not None:
    details = details[details["Prompt"] == prompt_type]

# Filter the exam
if exam_type is not None:
    details = details[details["Exam"] == exam_type]

# Filter out the comprehensive-100 exam
# details = details[details["Exam"] != "comprehensive-100"]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Answer Only", "Concise CoT", "Verbose CoT", "Composite"])

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Exam"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Get the model's color
colors = sns.color_palette()
color = colors[0] if model_type == "gpt-3.5" else colors[1]

# Plot the accuracy by prompt
plt.figure(figsize=(16, 9))
sns.barplot(
    x="Exam",
    y="Accuracy",
    data=results,
    color=color)
plt.title("Accuracy by Exam")
plt.xlabel("Exam")
plt.ylabel("Accuracy")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{output_folder}/accuracy-by-exam.png")
plt.show()