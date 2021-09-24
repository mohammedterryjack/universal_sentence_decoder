from typing import List, Tuple
from enum import Enum

from json import loads

class SquadDataKeys(Enum):
    DATA = "data"
    TITLE = "title"
    PARAGRAPH = "paragraphs"
    CONTEXT = "context"
    QUESTIONANSWER = "qas"
    QUESTION = "question"

def load_squad_data(path_to_file:str) -> List[Tuple[str,str]]:
    """
    formats the squad2 data
    (download from: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
    into a list of (context,question) pairs
    """
    with open(path_to_file) as json_file:
        squad2_data = loads(json_file.read())
        for topic_data in squad2_data[SquadDataKeys.DATA.value]:
            short_context = topic_data[SquadDataKeys.TITLE.value]
            for paragraph in topic_data[SquadDataKeys.PARAGRAPH.value]:
                long_context = paragraph[SquadDataKeys.CONTEXT.value]
                for index,question_data in enumerate(paragraph[SquadDataKeys.QUESTIONANSWER.value]):
                    question = question_data[SquadDataKeys.QUESTION.value]
                    if index == 0: yield (short_context, question)
                    yield (long_context, question)