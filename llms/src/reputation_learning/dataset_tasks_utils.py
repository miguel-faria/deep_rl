import json
import pandas as pd
import re

from pathlib import Path


class Dataset:
    def __init__(self, data_dir: Path, train_filename: str, test_filename: str, validation_filename: str):
        self._data_dir = data_dir
        self._train_file = train_filename
        self._validation_file = validation_filename
        self._test_file = test_filename

    @staticmethod
    def get_samples(file_path: Path):
        raise NotImplementedError('This method needs to be implemented by the class specific for the dataset to be used.')

    def get_train_samples(self):
        assert self._train_file != '', 'Training data file cannot be empty.'
        train_path = self._data_dir / self._train_file
        assert train_path.exists(), 'Training data path does not exist.'
        return self.get_samples(train_path)

    def get_test_samples(self):
        assert self._test_file != '', 'Testing data file cannot be empty.'
        test_path = self._data_dir / self._test_file
        assert test_path.exists(), 'Testing data path does not exist.'
        return self.get_samples(test_path)

    def get_validation_samples(self):
        assert self._validation_file != '', 'Validation data file cannot be empty.'
        validation_path = self._data_dir / self._validation_file
        assert validation_path.exists(), 'Validation data path does not exist.'
        return self.get_samples(validation_path)


class StrategyQA(Dataset):

    @staticmethod
    def get_samples(file_path: Path) -> pd.DataFrame:
        samples = pd.DataFrame()
        with (open(file_path, "r", encoding="utf-8-sig") as f):
            data = pd.read_json(f)
            for row in data.iterrows():
                row_data = pd.DataFrame({"id": [row[1]["qid"]], "question": [row[1]["question"]], "answer": ['yes'] if row[1]["answer"] else ['no'],
                                         "explanation": [' '.join(row[1]["facts"])]})
                samples = pd.concat([samples, row_data], ignore_index=True)

        return samples


class GSM8k(Dataset):

    @staticmethod
    def get_samples(file_path: Path) -> pd.DataFrame:
        samples = pd.DataFrame()
        with open(file_path, "r") as f:
            jsonlines = f.read().splitlines()
            for i, jsonline in enumerate(jsonlines):
                sample = json.loads(jsonline)
                answer = re.sub(r"[^0-9.]", "", sample["answer"].split("#### ")[1].strip())
                explanation = re.sub('<<.*>>', '', sample["answer"].split("#### ")[0].replace("\n\n", "\n").strip())
                explanation_sents = explanation.split("\n")
                explanation_sents = [explanation_sent + "." if explanation_sent[-1] != "." else explanation_sent for explanation_sent in explanation_sents]
                explanation = " ".join(explanation_sents)
                row_data = pd.DataFrame({"question": [sample["question"]], "answer": [answer], "explanation": [explanation]})
                samples = pd.concat([samples, row_data], ignore_index=True)

        return samples


class ECQA(Dataset):

    @staticmethod
    def get_samples(file_path: Path) -> pd.DataFrame:
        samples = pd.DataFrame()
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            options = [row["q_op1"], row["q_op2"], row["q_op3"], row["q_op4"], row["q_op5"]]
            row_data = pd.DataFrame({"question": [row["q_text"]], "options": [options], "answer": [str(options.index(row["q_ans"]) + 1)], "explanation": [row["taskB"]]})
            samples = pd.concat([samples, row_data], ignore_index=True)

        return samples
