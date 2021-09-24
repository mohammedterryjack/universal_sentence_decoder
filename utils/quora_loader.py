from typing import List

from pandas import read_csv

def load_quora_data(path_to_file:str) -> List[str]:
    """
    formats the Quora question data 
    (download from: https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv)
    into a single long list of unique questions
    """
    quora_dataset = read_csv(path_to_file, sep='\t', header=0)
    questions = quora_dataset.question1.append(quora_dataset.question2)
    questions = questions.apply(str)
    return sorted(set(questions))
