import json
import dropbox
import greedy_decoder
import summarizer
import data_handler


loader = data_handler.Loader()
loader.load_file("cnn_dailymail_v11_test.txt")
dbx = dropbox.Dropbox('YOUR_ACCESS_TOKEN')

model = summarizer.my_transformer_lm(mode='eval')
model.init_from_file("modelc1t150k.pkl")

result_list = []
for i, element in enumerate(loader.data['test']):
  prediction = greedy_decoder.greedy_predict(element['article'], model)
  result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})

  if i % 10 == 0:
      f = open("model_output_c1t150k", "w")
      f.write(json.dumps(result_list))
      f.close()
  if i % 200 == 0:
      name = "model_output_c1t150k" + str(i)
      path = name + ".txt"
      dbx.files_upload(name, path)