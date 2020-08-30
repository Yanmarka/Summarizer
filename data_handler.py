import json
import math
import random
import tensorflow_datasets as tfds
import numpy as np

class Loader():
  def __init__(self):
      self.data = {}
<<<<<<< HEAD
      self.question_index = {'train': 0, 'validation': 0, 'test': 0,}

  def load_data_from_tf(self, dataset='cnn_dailymail:3.0.0', long_text='article', short_text='highlights'):
    ds_train = tfds.load(dataset, split='train', shuffle_files=True)
    ds_test = tfds.load(dataset, split='test', shuffle_files=True)
    ds_validation = tfds.load(dataset, split='validation', shuffle_files=True)
    train_data, test_data, validation_data = [], [], []
=======
      self.question_index = 0
>>>>>>> 8897d9f798d7bbbef2304a8a1b8e4d41dec0a308

    for example in tfds.as_numpy(ds_train):
        train_data.append({'article':example[long_text].decode('utf-8'), 'highlights':example[short_text].decode('utf-8')})
    for example in tfds.as_numpy(ds_test):
        test_data.append({'article':example[long_text].decode('utf-8'), 'highlights':example[short_text].decode('utf-8')})
    for example in tfds.as_numpy(ds_validation):
      validation_data.append({'article':example[long_text].decode('utf-8'), 'highlights':example[short_text].decode('utf-8')})
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

  def load_next_question(self, ds='train'):
    article, summary = self.data[ds][self.question_index[ds]]['article'], self.data[ds][self.question_index[ds]]['highlights']
    self.question_index[ds] += 1
    if self.question_index[ds] >= len(self.data[ds]):
      self.question_index[ds] = 0
    return article, summary

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
        if len(example['highlights']) > length:
          print("Removed outlier where summary larger than limit") #There is one outlier where this happens
          continue
        difference = (len(example['article']) + len(example['highlights'])) - length
        ds_copy.append({'article': example['article'][difference:], 'highlights': example['highlights']}) 
      else:
        ds_copy.append(example)
    self.data[ds] = ds_copy

  def write_ids(self, TOKENIZER=None):
    enc_dict = {'train':[], 'test':[], 'validation': []}
    for example in self.data['train']:
      enc_dict['train'].append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})
    for example in self.data['test']:
      enc_dict['test'].append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})
    for example in self.data['validation']:
      enc_dict['validation'].append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})
    self.data = enc_dict
