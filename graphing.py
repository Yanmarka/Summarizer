import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import data_handler

base_path = "/home/yannick/Dropbox/Universität/Bachelorarbeit/Trained Models/"
configs = ["C1T1", "C1T2", "C1T3", "C1R1", "C1R2", "C1R3"]
configs2 = ["C2T1", "C2T2", "C2T3", "C2R1", "C2R2", "C2R3"]
c1_data = []
for element in configs:
    c1_data.append(base_path + element + "/log.txt")

def accuracy_graph(path, start=3):
    f = open(path)
    text = f.readlines()
    text = text[start:]
    acc = []
    for i in range(len(text)):
        if i % 14 == 0:
            acc.append(float(text[i].split()[-1]))
        if len(acc) >= 251:
            break
    f.close()
    return acc

def accuracy_graph_2(path, start=27):
    f = open(path)
    text = f.read().split()
    text = text[start:]
    acc = []
    for i in range(len(text)):
        if i % 37 == 0:
            acc.append(float(text[i][:-5]))
    f.close()
    return acc

def moving_average(numbers, window_size=3):
    i = 0
    moving_averages = []
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

def plot_graph(paths, colors=None):
    plot_data =  [moving_average(accuracy_graph(path, start=8)) for path in paths]
    if colors != None:
        for i, element in enumerate(plot_data):
            plt.plot(element, colors[i])
    else:
        for element in plot_data:
            plt.plot(element)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

def plot_scores(scores):
    for i, score in enumerate(scores):
        plt.plot(i, score['f'], 'bo')
        plt.plot(i, score['r'], 'ro')
        plt.plot(i, score['p'], 'go')

    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["T1", "R1", "T2", "R2", "T3", "R3", "T4", "R4"])

    red_patch = mpatches.Patch(color='red', label='Recall')
    blue_patch = mpatches.Patch(color='blue', label='F-Score')
    green_patch = mpatches.Patch(color='green', label='Precision')
    plt.legend(handles=[blue_patch, red_patch, green_patch])

def plot_dataset(length_list, text='article'):
    plt.boxplot(length_list, sym='')
    plt.ylabel("Text length in subword units")
    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        top=False,         
        labelbottom=False) 

def bar_graph(length_list, values):
    def make_labels(values):
        labels = []
        labels.append ("<" + str(values[1]))
        for i in range(len(values[2:])):
            labels.append(str(values[i-1] + "-" + str(values[i])))
        labels.append(">" + str(values[-1]))
        return labels

    buckets = []
    labels = make_labels(values)

    for i in range(len(values)):
        counter = 0
        for length_element in length_list:
            if values[i] > length_element > values[i-1]:
                counter += 1
        buckets.append(counter)
    buckets.append(len(length_list) - sum(buckets))

    plt.bar([1,2,3,4,5], buckets[1:], tick_label=labels)

    plt.xlabel("Length in subword units")
    plt.ylabel("Number of samples in dataset")

if __name__ == '__main__':
    loader = data_handler.Loader()
    loader.load_file("scientific_papers_16k.txt")
    loader.remove_long_examples(6000)
    preprocessor = data_handler.Preprocessor()
    length_merger = preprocessor.record_length(loader.data['train'], sections=['article'])

    loader = data_handler.Loader()
    loader.load_file("scientific_papers_16k_2.txt")
    loader.remove_long_examples(6000)
    length_merger += preprocessor.record_length(loader.data['train'], sections=['article'])
    
    bar_graph(length_merger, [0, 1000, 2000, 3000, 4000, 5000])
    plt.savefig("barchart_scientific_paper_articles.pdf")