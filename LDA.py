
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from utility.DBUtility import DBUtility
from argparse import ArgumentParser
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, time, pyLDAvis, pyLDAvis.sklearn, os

parser = ArgumentParser()

parser.add_argument("--model", help="LDA Model Path", dest="model")
parser.add_argument("--data", help="Data Path", dest="data", required=True)
parser.add_argument("-n", help="Keywords Number", dest="keyword", default=15, type=int)

MODELEXISTS = True 

base_path = os.path.dirname(os.path.abspath(__file__))
stpwrdpath = "stop_word_all.txt"

STOPWORDS = list()
n_components = range(50, 105, 10)
learning_decay = .7

with open(stpwrdpath, 'r', encoding = "utf-8-sig") as stopwords :
    for word in stopwords :
        STOPWORDS.append(word.replace("\n", ""))

def Get_Data_Vectors(contents):

    cntVector = CountVectorizer(stop_words = STOPWORDS)

    cntTf = cntVector.fit_transform(contents)

    return cntTf, cntVector

def Get_LDA_BestModel(cntTf, model_path):
    

    if model_path is not None:

        file = open(model_path, 'rb')

        best_model = pickle.load(file)

        file.close()

    else:

        perplexities = list()

        for i in n_components:

            lda = LatentDirichletAllocation(n_components = i, verbose = 10, learning_decay = learning_decay, n_jobs = 1)

            t0 = time.time()

            lda.fit(cntTf)
        
            gamma = lda.transform(cntTf)

            perplexity = lda.perplexity(cntTf, gamma)

            print ("Spent %.3f sec, Perplexity = %.3f"%(time.time() - t0, perplexity))

            perplexities.append(perplexity)
        
        res = pd.DataFrame({
            "perplexity": perplexities
        }, index=n_components)

        Get_LDA_Model_Performance(res)

        sorted_perplexities = np.argsort(perplexities)

        best_n = n_components[sorted_perplexities[0]]

        print("Best n_components: " + str(best_n))

        print ("Building Best LDA...")

        best_model = LatentDirichletAllocation(n_topics = best_n, verbose = 10, learning_decay = .7, n_jobs = 1)

        best_model.fit(cntTf)

        file = open('best_model.pickle', 'wb')

        pickle.dump(best_model, file)

        file.close()

    return best_model

def Get_LDA_Model_Performance(res):
    

    # Show graph
    ax = res.plot(figsize=(18, 9), marker=".")
    ax.margins(1, 0.5)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)

    ax.set_xlabel("n_components")
    ax.set_ylabel("perplexity")

    plt.title("Choosing Optimal LDA Model")
    plt.subplots_adjust(bottom=0.2)

    plt.savefig("performances.png")

def LDA_Distribution(model, cntTf, ids):
    
    print ("Caculating LDA Distribution...")

    lda_output = model.transform(cntTf)

    topicnames = ["Topic" + str(i) for i in range(model.n_topics)]

    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topicnames, index = ids)
    
    df_document_topic.to_csv("LDA_Distribution.csv", encoding = "utf-8")

def Get_MetaData(df):

    df = df.drop(columns = ["segs"])

    df["content"] = df["content"].str.replace("\n", "").str.replace("\t", "")

    df.to_csv("metadata.csv", encoding = "utf-8")

def Get_Keywords_Distribution(model, words, n_words):
    
    print ("Extracting Keywords Distribution...")

    topicnames = ["Topic" + str(i) for i in range(model.n_topics)]
    
    keywords = ["Keyword" + str(i) for i in range(1, 16)]

    topic_keywords = []

    for topic_weights in model.components_:

        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        res = list()
        _sum = np.sum(topic_weights)
        for i in top_keyword_locs:
            res.append(str(words[i]) + "/" + str(round(topic_weights[i]/_sum, 3)))

        topic_keywords.append(res)

    df_topic_keywords = pd.DataFrame(topic_keywords)

    df_topic_keywords.columns = keywords
    
    df_topic_keywords.index = topicnames

    df_topic_keywords.to_csv("keywords_distribution.csv", encoding = "utf-8")

def Model_Visualize(best_model, cntTf, cntVector):

    print ("Model Visualizing...")

    panel = pyLDAvis.sklearn.prepare(best_model, cntTf, cntVector, mds='tsne')

    pyLDAvis.save_html(panel, 'lda.html')

if __name__ == "__main__":
    
    args = parser.parse_args()

    model_path = args.model

    data_path = args.data
    
    n_words = args.keyword

    articles_df = pd.read_csv(os.path.join(base_path, data_path), encoding = "utf-8")

    articles_df["segs"] = articles_df["segs"].apply(eval).apply(" ".join)

    contents = list(articles_df["segs"])

    ids = list(articles_df["_id"])

    Get_MetaData(articles_df)

    cntTf, cntVector = Get_Data_Vectors(contents)

    words = cntVector.get_feature_names()
    
    best_model = Get_LDA_BestModel(cntTf, model_path)

    LDA_Distribution(best_model, cntTf, ids)

    Get_Keywords_Distribution(best_model, words, n_words)

    Model_Visualize(best_model, cntTf, cntVector)
	
