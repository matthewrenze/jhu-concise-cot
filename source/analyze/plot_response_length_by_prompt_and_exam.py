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
details = details[details["Prompt"] != "Composite"]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Answer Only", "Concise CoT", "Verbose CoT"])

# Calculate average output tokens
details["Output Tokens"] = details["Output Tokens"] / 10

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Prompt", "Exam"]) \
    .agg({"Output Tokens": "mean"}) \
    .reset_index()

# Create the FacetGrid for the small multiples
g = sns.FacetGrid(
    results,
    col="Exam",
    col_wrap=3,
    height=2,
    aspect=1.5)

# Define the plotting function for the FacetGrid
def plot_barplot(*args, **kwargs):
    data = kwargs.pop("data")
    sns.barplot(
        x="Prompt",
        y="Output Tokens",
        data=data,
        color=color)
    plt.xticks(rotation=15, ha="right")

# Plot the accuracy by temperature and prompt
g.map_dataframe(plot_barplot)
g.fig.suptitle(f"Response Length by Prompt and Exam for {model_type.upper()}", size=11)
g.fig.subplots_adjust(top=0.92)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Prompt", "Output Tokens")
g.savefig(f"{output_folder}/response-length-by-prompt-and-exam-for-{model_type}.png")
plt.show()