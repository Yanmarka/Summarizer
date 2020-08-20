from rouge_score import rouge_scorer
from sentencepiece import SentencePieceProcessor
import json
import numpy as np


model = "cnnd16k.model"
TOKENIZER = SentencePieceProcessor()
TOKENIZER.load(model)

def compute_rouge(path="model_output.txt"):
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    fscore = 0
    precision = 0
    recall = 0

    with open(path, "r") as f:
        testing_results = json.load(f)

    for i, element in enumerate(testing_results):
        scores = scorer.score(TOKENIZER.DecodeIds(element['prediction']), TOKENIZER.DecodeIds(element['reference']))
        precision += scores["rouge1"][0]
        recall += scores["rouge1"][1]
        fscore += scores["rouge1"][2]

    fscore = fscore / i
    precision = precision / i
    recall = recall / i

    return recall, precision, fscore

def compute_rouge_data(path="model_output.txt"):
    scorer = rouge_scorer.RougeScorer(['rouge1'])
    fscore = []
    precision = []
    recall = []

    with open(path, "r") as f:
        testing_results = json.load(f)

    for i, element in enumerate(testing_results):
        scores = scorer.score(TOKENIZER.DecodeIds(element['prediction']), TOKENIZER.DecodeIds(element['reference']))
        precision.append(scores["rouge1"][0])
        recall.append(scores["rouge1"][1])
        fscore.append(scores["rouge1"][2])

    return {'Max': {'Recall': (max(recall), np.argmax(recall)), 'Precision': (max(precision), np.argmax(precision)), 'Fscore': (max(fscore), np.argmax(fscore))}, 
            'Min': {'Recall': (min(recall), np.argmin(recall)), 'Precision': (min(precision), np.argmin(precision)), 'Fscore': (min(fscore), np.argmin(fscore))}}

def compute_time(path):
    total = 0
    k = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i % 14 == 0:
                total += float(line.split()[-2])
                k += 1
    return total / k

def merge_files(file1, file2, output_file):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    of = open(output_file, 'w')

    data = json.loads(f1.read()) + json.loads(f2.read())
    of.write(json.dumps(data))


if __name__ == "__main__":
    compute_rouge()
