import re

def rewrite_input(path="input.txt"):
    output_path = "split_" + path 
    with open(path, "r") as f:
        text = f.read()
    text = re.split(r'([\.\?\!])', text)
    text = [x+y for x,y in zip(text[0::2], text[1::2])]

    with open(output_path, "w") as f:
        for sentence in text:
            f.write("%s\n" % sentence)

def rewrite_vocab(path="vocab.vocab"):
    with open(path, "r") as f:
        text = f.readlines()
        cleaned_text = []
        for element in text:
            cleaned_text.append(re.split(r'\t+', element)[0])

    with open("model.vocab", "w") as f:
        i = 0
        for element in cleaned_text:
            f.write(element + "\t" + str(i) + "\n")
            i += 1
