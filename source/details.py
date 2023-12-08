import pandas as pd
import csv
from models.response import Response
from logs import Log


class DetailsRow():
    def __init__(self):
        self.id = 0
        self.date_time = ""
        self.model = ""
        self.prompt = ""
        self.exam = ""
        self.topic = ""
        self.source = ""
        self.source_id = 0
        self.problem_id = 0
        self.problem = ""
        self.dialog = ""
        self.answer_1 = ""
        self.answer_2 = ""
        self.answer_3 = ""
        self.answer_4 = ""
        self.answer_5 = ""
        self.answer_6 = ""
        self.answer_7 = ""
        self.answer_8 = ""
        self.answer_9 = ""
        self.answer_10 = ""
        self.agent_answer = ""
        self.correct_answer = ""
        self.solution = ""
        self.score = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.error = ""


class DetailsTable:

    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.table = pd.DataFrame()

    def create_header(self):
        column_names = DetailsRow().__dict__.keys()
        column_names = [column_name.replace("_", " ").title() for column_name in column_names]
        self.writer.writerow(column_names)

    def create_row(self):
        return DetailsRow()

    def add_row(self, row):
        self.table = self.table._append(row.__dict__, ignore_index=True)
        self.writer.writerow(row.__dict__.values())

    def close(self):
        self.file.close()


def update_details_row(details_row: DetailsRow, response: Response) -> None:

    # Update the tokens
    details_row.input_tokens += response.input_tokens
    details_row.output_tokens += response.output_tokens
    details_row.total_tokens += response.total_tokens

    # Handle errors
    if (response.has_error):
        details_row.error += response.text


def log_detail_summary(log: Log, details_row: DetailsRow):

    # Get the votes
    votes = []
    for i in range(1, 11):
        vote = getattr(details_row, f"answer_{i}")
        if vote is None:
            vote = "[None]"
        votes.append(vote)

    # Log the details
    log.info(f"Votes: {', '.join(votes)}")
    log.info(f"Agent Answer: {details_row.agent_answer}")
    log.info(f"Correct Answer: {details_row.correct_answer}")
    log.info(f"Score: {details_row.score}")
    log.info(f"Tokens: {details_row.total_tokens}")
    log.info("")



