from rouge_score import rouge_scorer
from sentencepiece import SentencePieceProcessor
import json


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
        fscore += scores[0]["rouge-1"][0]
        precision += scores[0]["rouge-1"][1]
        recall += scores[0]["rouge-1"][2]



    fscore = fscore / i
    precision = precision / i
    recall = recall / i

    print("FScore: " + str(fscore))
    print("Precision: " + str(precision))
    print("Recall: "+ str(recall))

def compute_time(path):
    total = 0
    k = 0
    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i % 14 == 0:
                total += float(line.split()[-2])
                k += 1
    return total / k


if __name__ == "__main__":
    compute_rouge()