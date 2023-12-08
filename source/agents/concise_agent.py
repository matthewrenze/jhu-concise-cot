from agents.agent import Agent

system_prompt = """
You are an intelligent assistant.
Your task is to answer the following multiple-choice questions.
Think step-by-step through the problem to ensure you have the correct answer.
Then, answer the question using the following format 'Action: Answer("[choice]")'  
The parameter [choice] is the letter or number of the answer you want to select (e.g. "A", "B", "C", or "D")
For example, 'Answer("C")' will select choice "C" as the best answer.
You MUST select one of the available choices; the answer CANNOT be "None of the Above".
Be concise.
"""

example_problem = """
Question: What is the capital of the state where Johns Hopkins University is located?
Choices:
  A: Baltimore
  B: Annapolis
  C: Des Moines
  D: Las Vegas
"""

example_solution = """
Thought: 
  Johns Hopkins University is located in Baltimore, Maryland.
  The capital of Maryland is Annapolis.
Action: Answer("B")  
"""

class ConciseAgent(Agent):

    def __init__(self, model, expertise, num_choices, log):
        super().__init__(model, expertise, num_choices, log)
        self.system_prompt_template = system_prompt
        self.example_problem = example_problem
        self.example_solution = example_solution
