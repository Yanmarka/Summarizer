from rouge import Rouge 
from sentencepiece import SentencePieceProcessor
import json

rouge = Rouge()

model = "cnnd16k.model"
TOKENIZER = SentencePieceProcessor()
TOKENIZER.load(model)

def compute_rouge(path="model_output.txt"):
    fscore = 0
    precision = 0
    recall = 0

    with open(path, "r") as f:
        testing_results = json.load(f)

    for i, element in enumerate(testing_results):
        try:
            scores = rouge.get_scores(TOKENIZER.DecodeIds(element['prediction']), TOKENIZER.DecodeIds(element['reference']))
            fscore += scores[0]["rouge-1"]["f"]
            precision += scores[0]["rouge-1"]["p"]
            recall += scores[0]["rouge-1"]["r"]
        except RecursionError:
            print("Error")


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