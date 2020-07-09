from sentencepiece import SentencePieceProcessor
import numpy as np
import data_handler

TOKENIZER = SentencePieceProcessor()
TOKENIZER.load('vocab.model')

loader = data_handler.Loader()
loader.load_file()

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
      article_enc = TOKENIZER.EncodeAsIds(article) + [1]
      summary_enc = TOKENIZER.EncodeAsIds(summary) + [1]
      combination = article_enc + [0] + summary_enc
      padding = create_padding(combination)
      values.append(np.concatenate((combination, padding)))
      mask.append(np.concatenate((np.zeros_like(article_enc), [0], np.ones_like(summary_enc), np.zeros_like(padding))))

    values = np.array(values)
    mask = np.array(mask)
    yield (values, values, mask)
