import json
import math
import random
import tensorflow_datasets as tfds
import numpy as np

class Loader():
  def __init__(self):
      self.data = {}

  def load_data_from_tf(self):
    ds_train = tfds.load('cnn_dailymail:3.0.0', split='train', shuffle_files=True)
    ds_test = tfds.load('cnn_dailymail:3.0.0', split='test', shuffle_files=True)
    ds_validation = tfds.load('cnn_dailymail:3.0.0', split='validation', shuffle_files=True)
    train_data = []
    test_data = []
    validation_data = []
    for example in tfds.as_numpy(ds_train):
        train_data.append({'article':example['article'].decode('utf-8'), 'highlights':example['highlights'].decode('utf-8')})
    for example in tfds.as_numpy(ds_test):
        test_data.append({'article':example['article'].decode('utf-8'), 'highlights':example['highlights'].decode('utf-8')})
    for example in tfds.as_numpy(ds_validation):
      validation_data.append({'article':example['article'].decode('utf-8'), 'highlights':example['highlights'].decode('utf-8')})
    self.data = {'train': train_data, 'test': test_data, 'validation': validation_data}

  def write_to_file(self, path='cnn_dailymail.txt', subset='all'):
      with open(path, 'w') as f:
          if subset=='all':
            f.write(json.dumps(self.data))
          else:
            f.write(json.dumps(self.data[subset]))

  def load_file(self, path='cnn_dailymail.txt'):  
      with open(path, 'r') as f:
          data = json.loads(f.read())
          self.data = data

  def load_random_question(self, ds='train'):
    number = np.random.randint(0, len(self.data[ds])) #TODO Should it be -1?
    return self.data[ds][number]['article'], self.data[ds][number]['highlights']
   
  def convert_to_textfile(self, path="input.txt"):
      subset_length = math.floor(len(self.data['train'])/3)
      training_data = self.data['train']
      random.shuffle(training_data)
      with open(path, "w") as f:
        for example in training_data[:subset_length]:
            f.write(example['article'])
            f.write(example['highlights'])
  
  def remove_long_examples(self, length=2045, ds='train'):
    filter_list = []
    for example in self.data[ds]:
      if len(example['article']) + len(example['highlights']) > length:
        filter_list.append(False)
      else:
        filter_list.append(True)
    data_array = np.array(self.data[ds])
    filtered_array = data_array[np.array(filter_list)]
    self.data[ds] = filtered_array.tolist()

  def shorten_long_examples(self, length=2045, ds='train'):
    ds_copy = []
    for example in self.data[ds]:
      if len(example['article']) + len(example['highlights']) > length:
        difference = (len(example['article']) + len(example['highlights'])) - length
        ds_copy.append({'article': example['article'][difference:], 'highlights': example['highlights']}) 
      else:
        ds_copy.append(example)
    self.data[ds] = ds_copy

  def write_ids(self, TOKENIZER=None):
    enc_dict = {'train':[], 'test':[]}
    for example in self.data['train']:
      enc_dict['train'].append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})
    for example in self.data['test']:
      enc_dict['test'].append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})
    self.data = enc_dict
