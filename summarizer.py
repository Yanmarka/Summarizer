from sentencepiece import SentencePieceProcessor
import numpy as np
from data_handler import Loader
import trax
import json
import configurations
from jax.config import config
import gin

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

def lm_input_function_eval(n_devices):
  batch_size = 8
  while True:
    values = []
    mask = []

    for i in range(n_devices*batch_size):
      article, summary = loader.load_next_question(ds='validation')
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

def configure_lsh(n_hashes):
  config_list = ["trax.layers.LSHSelfAttention.n_hashes = " + str(n_hashes),
  "trax.layers.LSHSelfAttention.chunk_len = 64"]
  gin.parse_config(config_list)

if __name__ == "__main__":
    FILE_PATH = "cnn_dailymail_v11.txt"
    OUTPUT_DIR = "./output_dir/"
    USE_TPU = False
    TPU_IP_ADRESS = None
    USE_LSH = False
    LSH_HASHES = 8
    EPOCHS = 1

    if USE_TPU:
      backend_target = "grpc://" + TPU_IP_ADRESS + ":8470"
      
      config.FLAGS.jax_xla_backend = "tpu_driver"
      config.FLAGS.jax_backend_target = backend_target

    if USE_LSH:
      configure_lsh(LSH_HASHES)

    loader = Loader()
    loader.load_file(FILE_PATH)

    #model = configurations.lsh_reformer(mode='train')
    summarize_inputs = trax.supervised.inputs.Inputs(train_stream=lm_input_function, eval_stream=lm_input_function_eval)

    trainer = trax.supervised.Trainer(
      model=configurations.c1_reformer,
      loss_fn=trax.layers.CrossEntropyLoss(),
      optimizer=trax.optimizers.Adafactor,
      lr_schedule=trax.lr.MultifactorSchedule,
      inputs=summarize_inputs,
      output_dir=OUTPUT_DIR)

    for i in range(EPOCHS):
      trainer.train_epoch(200,10)