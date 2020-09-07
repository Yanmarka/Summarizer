import json
import math
import random
import tensorflow_datasets as tfds
import numpy as np
import re

class Loader():
  def __init__(self):
      self.data = {}
      self.question_index = {'train': 0, 'validation': 0, 'test': 0,}

  def load_data_from_tf(self, dataset='cnn_dailymail:3.0.0', long_text='article', short_text='highlights'):
    ds_train = tfds.load(dataset, split='train', shuffle_files=True)
    ds_test = tfds.load(dataset, split='test', shuffle_files=True)
    ds_validation = tfds.load(dataset, split='validation', shuffle_files=True)
    train_data, test_data, validation_data = [], [], []

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

  def write_ids(self, text, TOKENIZER=None, print_progress=False):
    ids = []
    for i, example in enumerate(text):
      ids.append({'article': TOKENIZER.EncodeAsIds(example['article']), 'highlights': TOKENIZER.EncodeAsIds(example['highlights'])})

      if print_progress and i % 1000 == 0:
        print("Completed " + str(i) + " of " + str(len(text)) + " samples.") 

    return ids

  def full_id_conversion(self, tokenizer):
    enc_dict = {'train':[], 'test':[], 'validation': []}
    enc_dict['train'] = self.write_ids(self.data['train'], tokenizer, True)
    enc_dict['test'] = self.write_ids(self.data['test'], tokenizer, True)
    enc_dict['validation'] = self.write_ids(self.data['validation'], tokenizer, True)

    self.data = enc_dict

class Preprocessor():
  def rewrite_input(self, path="input.txt"):
    output_path = "split_" + path 
    with open(path, "r") as f:
        text = f.read()
    text = re.split(r'([\.\?\!])', text)
    text = [x+y for x,y in zip(text[0::2], text[1::2])]

    with open(output_path, "w") as f:
        for sentence in text:
            f.write("%s\n" % sentence)

  def rewrite_vocab(self, path="vocab.vocab"):
      with open(path, "r") as f:
          text = f.readlines()
          cleaned_text = []
          for element in text:
              cleaned_text.append(re.split(r'\t+', element)[0])

      with open(self, "model.vocab", "w") as f:
          i = 0
          for element in cleaned_text:
              f.write(element + "\t" + str(i) + "\n")
              i += 1

  def write_sentencepiece_input_file(self, data, path="sentencepiece_input.txt"):
      with open(path, 'w') as f:
          for document in data:
            for section in ['article', 'highlights']:
              for line in document[section].splitlines():
                f.write(line + '\n')

def load_scientific_papers_data():
  pubmed_loader = Loader()
  pubmed_loader.load_data_from_tf(dataset="scientific_papers/pubmed:1.1.1", short_text='abstract')
  arxiv_loader = Loader()
  arxiv_loader.load_data_from_tf(dataset="scientific_papers/arxiv:1.1.1", short_text='abstract')

  merged_training_data =  arxiv_loader.data['train']+  pubmed_loader.data['train']
  random.shuffle(merged_training_data)
  first_half = {}
  second_half = {}

  first_half['train'] = merged_training_data[math.floor(len(merged_training_data)/2):] 
  second_half['train'] = merged_training_data[:math.floor(len(merged_training_data)/2)] 
  first_half['test'] = pubmed_loader.data['test'] + arxiv_loader.data['test']
  second_half['test'] = pubmed_loader.data['test'] + arxiv_loader.data['test']
  first_half['validation'] = pubmed_loader.data['validation'] + arxiv_loader.data['validation']
  second_half['validation'] = pubmed_loader.data['validation'] + arxiv_loader.data['validation']

  loader = Loader()
  loader.data = first_half
  loader.write_to_file("scientific_papers_text_one.txt")

  loader.data = second_half
  loader.write_to_file("scientific_papers_text_two.txt")