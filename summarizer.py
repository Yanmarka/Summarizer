from sentencepiece import SentencePieceProcessor
import numpy as np
import data_handler
import trax

TOKENIZER = SentencePieceProcessor()
TOKENIZER.load('cnnd16k.model')

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