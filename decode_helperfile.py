import json
import greedy_decoder
import summarizer
import data_handler

def main(model_name, stop=5000):
    loader = data_handler.Loader()
    loader.load_file("cnn_dailymail_v11_validation.txt")

    model = summarizer.my_transformer_lm(mode='eval')
    model.init_from_file(model_name)

    path = model_name + "_output.txt"
    result_list = []
    for i, element in enumerate(loader.data['validation']):
        prediction = greedy_decoder.greedy_predict(element['article'], model)
        result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})

        if i % 10 == 0:
            f = open(path, "w")
            f.write(json.dumps(result_list))
            f.close()
            print(i)

        if i == stop:
            break
