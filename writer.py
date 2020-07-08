import re

def rewrite_input(path="input.txt"):
    output_path = "split_" + path 
    with open(path, "r") as f:
        text = f.read()
        text = text.split('.')

    with open(output_path, "w") as f:
        for sentence in text:
            f.write("%s\n" % sentence)

def rewrite_vocab(path="vocab.vocab"):
    with open("test.vocab", "r") as f:
        text = f.readlines()
        cleaned_text = []
        for element in text:
            cleaned_text.append(re.split(r'\t+', element)[0])

    with open("model.vocab", "w") as f:
        i = 1
        for element in cleaned_text:
            f.write(element + "\t" + str(i) + "\n")
            i += 1