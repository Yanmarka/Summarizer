import greedy_decoder
import data_handler
import summarizer
import json

model = summarizer.my_reformer_no_lsh(mode='eval')
model.init_from_file('model.pkl')
loader = data_handler.Loader()
loader.load_file(path="cnn_dailymail_v11_test.txt")
result_list =[]

for i, element in enumerate(loader.data['test']):
  
  prediction = greedy_decoder.greedy_predict(element['article'], model)
  result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})

  if i % 100 == 0:
    f = open("model_output_reformer.txt", "w")
    f.write(json.dumps(result_list))
    f.close()
  if i % 1000 == 0:
    exit()