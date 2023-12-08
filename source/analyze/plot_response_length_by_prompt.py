# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Set the parameters
model_type = "gpt-4"
prompt_type = None
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
details = details[details["Exam"] != "comprehensive-100"]

# Filter out the composite prompt
details = details[details["Prompt"] != "Answer Only"]
details = details[details["Prompt"] != "Composite"]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Concise CoT", "Verbose CoT"])

# Calculate average output tokens
details["Output Tokens"] = details["Output Tokens"] / 10

# Get the model's color
colors = sns.color_palette()
color = colors[0] if model_type == "gpt-3.5" else colors[1]

# Create a barplot of the average output tokens by prompt
plt.figure(figsize=(5, 5))
sns.boxplot(
    x="Prompt",
    y="Output Tokens",
    data=details,
    color=color)
plt.title(f"Response Length by Prompt for {model_type.upper()}")
plt.xlabel("Prompt")
plt.ylabel("Output Tokens")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, 350)
plt.tight_layout()
plt.savefig(f"{output_folder}/response-length-by-prompt-for-{model_type}.png")
plt.show()