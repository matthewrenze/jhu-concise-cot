# The Benefits of a Concise Chain of Thought on Problem-Solving in Large Language Models

**Author:** Matthew Renze  
**Class:** EN.605.801  
**Date:** 2023-12-08  

## Abstract
In this paper, we introduce Concise Chain-of-Thought (CCoT) prompting. We compared standard CoT and CCoT prompts to see how conciseness impacted response length and correct-answer accuracy. We evaluated this using GPT-3.5 and GPT-4 with a multiple-choice question-and-answer (MCQA) benchmark.

CCoT reduced average response length by 48.70% for both GPT-3.5 and GPT-4 while having a negligible impact on problem-solving performance. However, on math problems, GPT-3.5 with CCoT incurs a performance penalty of 27.69%. Overall, CCoT leads to an average per-token cost reduction of 22.67%.

These results have practical implications for AI systems engineers using LLMs to solve real-world problems with CoT prompt-engineering techniques. In addition, these results provide more general insight for AI researchers studying the emergent behavior of step-by-step reasoning in LLMs.

## Documents
- [Research paper](research-paper.pdf)
- [Technical appendix](technical-appendix.pdf)

## Code
- [Source](source/) - contains all source code
- [Models](source/models) - contains the model-specific code
- [Prompts](source/agents) - contains LLM agent prompt code
- [Process](source/process/) - contains the data pre-processing scripts
- [Analyze](source/analyze/) - contains the data analysis scripts

## Data
- [Exams](exams/) - contains the test dataset
- [Results](results/) - contains the high-level test results
- [Details](details/) - contains the low-level test results
- [Logs](logs/) - contains the experiment event logs


## Analysis
- [Plots](plots/) - contains all data visualizations

