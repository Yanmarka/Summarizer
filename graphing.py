import matplotlib.pyplot as plt

base_path = "/home/yannick/Dropbox/Universität/Bachelorarbeit/Trained Models/"
configs = ["C1T1", "C1T2", "C1T3", "C1R1", "C1R2", "C1R3"]
c1_data = []
for element in configs:
    c1_data.append(base_path + element + "/log.txt")

def accuracy_graph(path, start=2):
    f = open(path)
    text = f.readlines()
    text = text[start:]
    acc = []
    for i in range(len(text)):
        if i % 14 == 0:
            acc.append(float(text[i].split()[-1]))
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
    plot_data =  [moving_average(accuracy_graph(path)) for path in paths]
    if colors != None:
        for i, element in enumerate(plot_data):
            plt.plot(element, colors[i])
    else:
        for element in plot_data:
            plt.plot(element)
    plt.show()

if __name__ == '__main__':
    plot_graph(c1_data, ['#0000FF', '#00FFFF', '#1569C7', '#FF0000', '#DC143C', '#ff0081'])
