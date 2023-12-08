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
model_type = "gpt-3.5"
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

# Filter out the answer-only and composite prompt
details = details[details["Prompt"] != "Answer Only"]
details = details[details["Prompt"] != "Composite"]

# Sort the data by agent
details["Prompt"] = pd.Categorical(details["Prompt"], ["Concise CoT", "Verbose CoT"])

# Select only the problem ID and output tokens columns
details = details[["Prompt", "Problem Id", "Accuracy"]]

cot_data = details[details["Prompt"] == "Verbose CoT"]
ccot_data = details[details["Prompt"] == "Concise CoT"]

# Plot the distribution of the cot data
plt.figure(figsize=(10, 5))
sns.histplot(
    x="Accuracy",
    data=cot_data)
plt.title(f"Distribution of Accuracy for {model_type.upper()} with Verbose CoT")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_folder}/distribution-of-accuracy-for-{model_type}-with-vcot.png")
plt.show()

# Plot the distribution of the ccot data
plt.figure(figsize=(10, 5))
sns.histplot(
    x="Accuracy",
    data=ccot_data,
    color=sns.color_palette()[1])
plt.title(f"Distribution of Accuracy for {model_type.upper()} with Concise CoT")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_folder}/distribution-of-accuracy-for-{model_type}-with-ccot.png")
plt.show()

# Test normality
cot_normality = stats.shapiro(cot_data["Accuracy"])
ccot_normality = stats.shapiro(ccot_data["Accuracy"])
print(cot_normality)
print(ccot_normality)

# Test variance
variance = stats.levene(cot_data["Accuracy"], ccot_data["Accuracy"])
print(variance)

# Perform the t-test
result = stats.ttest_ind(cot_data["Accuracy"], ccot_data["Accuracy"], equal_var=False)
print(result)
print(f"t-statistic: {result.statistic:.2f}")
print(f"p-value: {result.pvalue}")
print(f"Degrees of freedom: {result.df:.2f}")

# Perform the Mann-Whitney U test
u_results = stats.mannwhitneyu(cot_data["Accuracy"], ccot_data["Accuracy"], alternative='two-sided')
print(f"U-statistic: {u_results.statistic:.2f}")
print(f"p-value: {u_results.pvalue:.2f}")

# Compute the percent decrease in output tokens from CoT to CCoT
cot_mean = cot_data["Accuracy"].mean()
ccot_mean = ccot_data["Accuracy"].mean()
percent_decrease = (cot_mean - ccot_mean) / cot_mean * 100
print(f"Decrease in accuracy: {percent_decrease:.2f}%")

# Get the model's color
colors = sns.color_palette()
color = colors[0] if model_type == "gpt-3.5" else colors[1]

# Create a barplot of the average output tokens by prompt
plt.figure(figsize=(5, 5))
sns.boxplot(
    x="Prompt",
    y="Accuracy",
    data=details,
    color=color,
    notch=True)
plt.title(f"Accuracy by Prompt for {model_type.upper()}")
plt.xlabel("Prompt")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{output_folder}/accuracy-by-prompt-for-{model_type}.png")
plt.show()

# Create the violin plot
plt.figure(figsize=(5, 5))
sns.violinplot(
    x="Prompt",
    y="Accuracy",
    data=details,
    color=color,
    inner=None)
plt.title(f"Accuracy by Prompt for GPT-4")
plt.xlabel("Prompt")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{output_folder}/violin-accuracy-by-prompt-for-gpt-4.png")
plt.show()
