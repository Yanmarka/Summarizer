from rouge import Rouge 
from sentencepiece import SentencePieceProcessor
import json

rouge = Rouge()

model = "cnnd16k.model"
TOKENIZER = SentencePieceProcessor()
TOKENIZER.load(model)

def compute_rouge(path="testing_results.txt"):
    fscore = 0
    precision = 0
    recall = 0

    with open(path, "r") as f:
        testing_results = json.load(f)

    for i, element in enumerate(testing_results):
        scores = rouge.get_scores(TOKENIZER.DecodeIds(element['hypothesis']), TOKENIZER.DecodeIds(element['reference']))
        fscore += scores[0]["rouge-1"]["f"]
        precision += scores[0]["rouge-1"]["p"]
        recak += scores[0]["rouge-1"]["r"]

    fscore = fscore / i
    precision = precision / i
    recall = recall / i

    print("FScore: " + str(fscore))
    print("Precision: " + str(precision))
    print("Recall: "+ str(recall))

if __name__ == "__main__":
    compute_rouge()