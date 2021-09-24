from typing import List, Tuple
from enum import Enum

class DailyDialogKeys(Enum):
    QUESTION_ACT = "2"
    END_OF_UTTERANCE = "__eou__"
    ACT_FILENAME = "dialogues_act.txt"
    DIALOGUE_FILENAME = "dialogues_text.txt"

def load_daily_dialog(path_to_file:str) -> List[Tuple[str,str]]:
    """
    formats the Daily Dialog data 
    (download from: https://aclanthology.org/attachments/I17-1099.Datasets.zip)
    into a list of (context,questions) pairs
    wherein the act (2) is used to identify questions
    and the conversation prior to it is used as the context
    (the remaining part of the diaogue is ignored
    as are dialogues without any questions in)
    """
    with open(f"{path_to_file}/{DailyDialogKeys.ACT_FILENAME.value}") as dialogue_act_file:
        dialogue_acts = dialogue_act_file.readlines()

    with open(f"{path_to_file}/{DailyDialogKeys.DIALOGUE_FILENAME.value}") as dialogue_file:
        dialogues = dialogue_file.readlines()

    for line_dialogue,line_acts in zip(dialogues,dialogue_acts):
        question_in_dialogue = DailyDialogKeys.QUESTION_ACT.value in line_acts[1:]
        if question_in_dialogue:
            utterances = line_dialogue.split(DailyDialogKeys.END_OF_UTTERANCE.value)[:-1]
            question_act_indexes = [
                position for position, act_index in enumerate(line_acts.strip().split()) \
                if position > 0 and act_index == DailyDialogKeys.QUESTION_ACT.value
            ]
            for question_index in question_act_indexes:
                question = utterances[question_index]
                context = utterances[:question_index]
                yield (' '.join(context),question)