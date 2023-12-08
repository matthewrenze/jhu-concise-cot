# Import the packages
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.patches as patches
from scipy import stats

# Set the parameters
model_type = "gpt-4"
input_file = f"../details/all-details.csv"
output_folder = f"../plots"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Load the data
details = pd.read_csv(input_file)

# Filter the model
details = details[details["Model"] == model_type]

# Filter out the comprehensive-100 exam
details = details[details["Exam"] != "comprehensive-100"]

# Filter out the composite prompt
details = details[details["Prompt"] != "Answer Only"]
details = details[details["Prompt"] != "Composite"]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Concise CoT", "Verbose CoT"])

# Calculate average output tokens (since there are 10 responses per problem)
details["Output Tokens"] = details["Output Tokens"] / 10

# Get the token cost per 1k tokens
input_token_cost_per_1k = 0.03 if model_type == "gpt-4" else 0.001
output_token_cost_per_1k = 0.06 if model_type == "gpt-4" else 0.002

# Calculate the token cost
details["Input Cost"] = details["Input Tokens"] / 1000 * input_token_cost_per_1k
details["Output Cost"] = details["Output Tokens"] / 1000 * output_token_cost_per_1k
details["Total Cost"] = details["Input Cost"] + details["Output Cost"]

# Select only the problem ID and output tokens columns
details = details[["Prompt", "Problem Id", "Input Cost", "Output Cost", "Total Cost"]]

cot_data = details[details["Prompt"] == "Verbose CoT"]
ccot_data = details[details["Prompt"] == "Concise CoT"]

print("Costs per 1k problems:")

# Calculate the average input cost per question
cot_input_cost = cot_data["Input Cost"].mean() * 1000
ccot_input_cost = ccot_data["Input Cost"].mean() * 1000
print(f"CoT input cost:  ${cot_input_cost:.2f}")
print(f"CCoT input cost: ${ccot_input_cost:.2f}")

# Calculate the average output cost per question
cot_output_cost = cot_data["Output Cost"].mean() * 1000
ccot_output_cost = ccot_data["Output Cost"].mean() * 1000
print(f"CoT output cost:  ${cot_output_cost:.2f}")
print(f"CCoT output cost: ${ccot_output_cost:.2f}")

# Calculate the average total cost per question
cot_total_cost = cot_data["Total Cost"].mean() * 1000
ccot_total_cost = ccot_data["Total Cost"].mean() * 1000
print(f"CoT total cost:  ${cot_total_cost:.2f}")
print(f"CCoT total cost: ${ccot_total_cost:.2f}")

# Calculate the total cost savings in % of CCoT over CoT
total_cost_savings = (cot_total_cost - ccot_total_cost) / cot_total_cost * 100
print(f"Total cost savings: {total_cost_savings:.2f}%")

# Get the model's color
colors = sns.color_palette()
color = colors[0] if model_type == "gpt-3.5" else colors[1]

# Create a barplot of the average output tokens by prompt
plt.figure(figsize=(5, 5))
sns.boxplot(
    x="Prompt",
    y="Total Cost",
    data=details,
    color=color,
    notch=True)
plt.title(f"Total Cost by Prompt for {model_type.upper()}")
plt.xlabel("Prompt")
plt.ylabel("Total Cost per Problem ($)")
plt.tight_layout()
plt.savefig(f"{output_folder}/total-cost-by-prompt-for-{model_type}.png")
plt.show()
