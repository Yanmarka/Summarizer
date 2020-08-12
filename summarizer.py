from sentencepiece import SentencePieceProcessor
import numpy as np
import data_handler
import trax
import json


def create_padding(text, amount=2048):
  if len(text) < amount:
    shape = amount-len(text)
    padding = np.zeros(shape, dtype=int)
    return np.concatenate((text, padding)), padding, True
  else:
    return np.array(text), None, False

def lm_input_function(n_devices):
  batch_size = 8
  while True:
    values = []
    mask = []

    for i in range(n_devices*batch_size):
      article, summary = loader.load_next_question()
      article = article + [1]
      summary = summary + [1]
      combination = article + [0] + summary
      padded_text, padding, is_padded = create_padding(combination)
      values.append(padded_text)
      if is_padded == True:
        mask.append(np.concatenate((np.zeros_like(article), [0], np.ones_like(summary), np.zeros_like(padding))))
      else:
        mask.append(np.concatenate((np.zeros_like(article), [0], np.ones_like(summary))))

    values = np.array(values)
    mask = np.array(mask)
    yield (values, values, mask)

def my_transformer_lm(mode):
  return trax.models.TransformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode)

def my_reformer_lm(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode, ff_activation=trax.layers.Relu, attention_type=trax.layers.LSHSelfAttention)

def my_reformer_no_lsh(mode):
  return trax.models.ReformerLM(vocab_size=16000, max_len=2048, d_model=1024, d_ff=4096, n_layers=3, mode=mode, ff_activation=trax.layers.Relu)

def create_trainer(model, inputs, output_dir):
  trainer = trax.supervised.Trainer(
      model=model,
      loss_fn=trax.layers.CrossEntropyLoss(),
      optimizer=trax.optimizers.Adafactor,
      lr_schedule=trax.lr.MultifactorSchedule,
      inputs=inputs,
      output_dir=output_dir)
  return trainer