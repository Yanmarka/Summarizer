import json
import greedy_decoder
import new_greedy_decoder
import data_handler

def main(model_name, model, start=-1, stop=None, dataset='cnn_dailymail_v11_validation.txt'):
    loader = data_handler.Loader()
    loader.load_file(dataset)

    model.init_from_file(model_name)

    path = model_name + "_output.txt"
    result_list = []
    for i, element in enumerate(loader.data['validation']):
        if i < start:
            continue

        prediction = greedy_decoder.greedy_predict(element['article'], model)
        result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})

        if i % 100 == 0:
            f = open(path, "w")
            f.write(json.dumps(result_list))
            f.close()
            print("Ran train step " + str(i) + " for model " + model_name)

        if i == stop:
            break
    
    f = open(path, "w")
    f.write(json.dumps(result_list))
    f.close()
    print("Ran complete dataset for model " + model_name + ". Done!")

def main2(model_name, model, start=-1, stop=None, dataset='scientific_papers_4k.txt'):
    loader = data_handler.Loader()
    loader.load_file(dataset)

    model.init_from_file(model_name)

    path = model_name + "_output.txt"
    result_list = []
    for i, element in enumerate(loader.data['validation']):
        if i < start:
            continue

        prediction = new_greedy_decoder.greedy_predict(element['article'], model)
        result_list.append({'article': element['article'], 'prediction': prediction, 'reference': element['highlights']})

        if i % 100 == 0:
            f = open(path, "w")
            f.write(json.dumps(result_list))
            f.close()
            print("Ran train step " + str(i) + " for model " + model_name)

        if i == stop:
            break
    
    f = open(path, "w")
    f.write(json.dumps(result_list))
    f.close()
    print("Ran complete dataset for model " + model_name + ". Done!")