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

# Group by prompt and temperature and average the accuracy
results = details \
    .groupby(["Prompt", "Exam"]) \
    .agg({"Accuracy": "mean"}) \
    .reset_index()

# Get the model's color
colors = sns.color_palette()
color = colors[0] if model_type == "gpt-3.5" else colors[1]

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
        y="Accuracy",
        data=data,
        color=color)
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")

# Plot the accuracy by temperature and prompt
g.map_dataframe(plot_barplot)
g.fig.suptitle(f"Accuracy of {model_type.upper()} by Prompt, and Exam", size=11)
g.fig.subplots_adjust(top=0.92)
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Prompt", "Accuracy")
g.savefig(f"{output_folder}/accuracy-by-prompt-and-exam.png")
plt.show()