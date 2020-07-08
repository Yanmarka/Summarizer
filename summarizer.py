import trax
import tensorflow_datasets as tfds
from sentencepiece import SentencePieceProcessor
import numpy as np
import time
import json

#TOKENIZER = SentencePieceProcessor()
#TOKENIZER.load('cp.320.model')

class Loader():
  def __init__(self):
      self.data = {}

  def load_data_from_tf(self):
    ds_train = tfds.load('cnn_dailymail', split='train')
    ds_test = tfds.load('cnn_dailymail', split='test')
    train_data = []
    test_data = []
    for example in tfds.as_numpy(ds_train):
        train_data.append({'article':example['article'].decode('utf-8'), 'highlights':example['highlights'].decode('utf-8')})
    for example in tfds.as_numpy(ds_test):
        test_data.append({'article':example['article'].decode('utf-8'), 'highlights':example['highlights'].decode('utf-8')})
    self.data = {'train': train_data, 'test': test_data}

  def write_to_file(self):  
      with open('cnn_dailymail.txt', 'w') as f:
          f.write(json.dumps(self.data))

  def load_file(self):  
      with open('cnn_dailymail.txt', 'r') as f:
          data = json.loads(f.read())
          self.data = data

  #def load_random_question(self):
    #number = np.random.randint(0, self.total_examples)
    #return self.examples[number]['article'], self.examples[number]['highlights']
   
          


def create_padding(text, amount=4096):
  if len(text) < amount:
    shape = amount-len(text)
    return np.zeros(shape, dtype=int)

def lm_input_function(n_devices):
  batch_size = 1
  while True:
    values = []
    mask = []

    for i in range(n_devices*batch_size):
      article, summary = loader.load_random_question()
      article_enc = TOKENIZER.EncodeAsIds(article)
      summary_enc = TOKENIZER.EncodeAsIds(summary)
      x = article_enc + [0] + summary_enc
      padding = create_padding(x)
      values.append(np.concatenate((x, padding)))
      mask.append(np.concatenate((np.zeros_like(article_enc), [0], np.ones_like(summary_enc), np.zeros_like(padding))))

    values = np.array(values)
    mask = np.array(mask)
    yield (values, values, mask)

#summarize_inputs = trax.supervised.inputs.Inputs(lm_input_function)

#s = time.time()
#x, _, m = next(summarize_inputs.train_stream(1))
#print(time.time()-s)
#print(*x[0])
#input()
#print(*m[0])

loader = Loader()
#loader.load_data_from_tf()
#loader.write_to_file()
#loader.load_file()
