import seaborn as sns
from data_handler import Loader
import matplotlib.pyplot as plt

def record_length(data, section):
    length = []
    for element in data:
        length.append(len(element[section]))
    return length

loader1 = Loader()
loader2 = Loader()
loader = Loader()

loader1.load_file("scientific_papers_16k.txt")
loader1.remove_long_examples(6000, 'train')

loader2.load_file("scientific_papers_16k_2.txt")
loader2.remove_long_examples(6000, 'train')

loader.load_file("cnn_dailymail_enc.txt")
loader1.data['train'] = loader1.data['train'] + loader2.data['train']
scientific_paper_articles = record_length(loader1.data['train'], 'article')
cnn_articles = record_length(loader.data['train'], 'article')

scientific_paper_summaries = record_length(loader1.data['train'], 'highlights')
cnn_summaries = record_length(loader.data['train'], 'highlights')

article_plot = sns.boxplot(data=[scientific_paper_articles, cnn_articles], showfliers=False)
article_plot.set(xticklabels=[])
article_figure = article_plot.get_figure()
article_figure.savefig("articles.pdf")
plt.clf()

summary_plot = sns.boxplot(data=[scientific_paper_summaries, cnn_summaries], showfliers=False)
summary_plot.set(xticklabels=[])
summary_figure = summary_plot.get_figure()
summary_figure.savefig("summaries.pdf")
