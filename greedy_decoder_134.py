from trax import layers as tl
from data_handler import Loader
import numpy as np
import json
import configurations
import gin

def create_padding(text, amount=2048):
  if len(text) < amount:
    shape = amount-len(text)
    padding = np.zeros(shape, dtype=int)
    return np.concatenate((text, padding)), padding, True
  else:
    return np.array(text), None, False

def generate_output(model, inputs, limit=155):
  counter = 0
  current_symbols = [0]
  sample = None
  while counter < limit and sample != [1]:
    logits = model((inputs, current_symbols))[0]
    sample = tl.logsoftmax_sample(logits[:, -1, :], temperature=0)
    current_symbols.append(sample)
    counter += 1
  return current_symbols

def run_evaluation(model, ds='validation', output_path="cnn_dailymail_134_output.txt"):
    result_list = []
    for i, element in enumerate(loader.data['ds']):
        article = element['article'] + [1]
        article = create_padding(np.array(article), 2048)[0]
        prediction = generate_output(model, article[None, :])
        result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})
        if i % 100 == 0:
            f = open(output_path, "w")
            f.write(json.dumps(result_list))
            f.close()
            print("Ran train step " + str(i)) #+ " for model " + model_name)

if __name__ == "__main__":
    loader = Loader()
    loader.load_file("cnn_dailymail_v11_validation.txt")

    gin.parse_config("""
    trax.layers.SelfAttention.chunk_len=64
    """)

    my_model =  configurations.c1_reformer(mode='predict')
    my_model.init_from_file("./model.pkl.gz",weights_only=True)
    run_evaluation(my_model, 'validation')