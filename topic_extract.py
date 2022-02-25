import pandas as pd
import random
import numpy as np
import os
from umap import UMAP
from bertopic import BERTopic
import jieba
import pickle


class TopicExtractor:
    def __init__(self, data_dir) -> None:
        """
        data_dir: Weibo Dataset Directory.
        """
        self.data_dir = data_dir
        self.csv_file = os.path.join(data_dir, "content.csv")
        self.topic_file = os.path.join(data_dir, "topic.pkl")
        self.content_topic_file = os.path.join(data_dir, "content_topic.csv")

    def extract(self, topic_num=None, random_seed=1000):
        df = pd.read_csv(self.csv_file)
        docs_raw = df['content'].tolist()

        # 如果对随机有要求，确定umap的随机种子
        if random_seed:
            np.random.seed(random_seed)
            umap_model = UMAP(n_neighbors=15, n_components=5, 
                            min_dist=0.0, metric='cosine', random_state=random_seed)
        else:
            umap_model = None

        # 文本处理，切词
        stoptext = open('stopwords.txt',encoding='utf-8').read()
        stopwords = [ item.strip() for item in stoptext.split('\n') ]
        def clean_text(text):
            words=jieba.lcut(text)
            words=[w for w in words if (w not in stopwords) and (not w.encode('UTF-8').isalnum())]
            return ' '.join(words)
        docs = [clean_text(doc) for doc in docs_raw]

        # 建立主题模型
        topic_model = BERTopic(language="chinese (simplified)", calculate_probabilities=True, umap_model=umap_model, verbose=True)
        topics, probs = topic_model.fit_transform(docs)

        freq = topic_model.get_topic_info(); 
        print("The original number of themes:", len(freq), "; The number of docs:", len(docs))

        if topic_num:
            topics, probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=topic_num)
        
        freq = topic_model.get_topic_info()
        print("Top 20 topics:")
        print(freq.head(20))

        print("Example: Topic 0: ")
        print(topic_model.get_topic(0))

        # save topic file
        topic_list = {}
        for topic_id in freq["Topic"].tolist():
            topic_ = topic_model.get_topic(topic_id)
            assert len(topic_) == 10
            topic_list[topic_id] = {"words": [word[0] for word in topic_], 
                "words_prob": np.array([word[1] for word in topic_]),
                "embedding": np.array(topic_model.topic_embeddings[topic_id+1])}
        with open(self.topic_file, 'wb') as f:
            pickle.dump(topic_list, f)

        # save topics of contents
        content_topics = []
        topn = 5
        for i in range(len(docs)):
            similar_topics, similarity = topic_model.find_topics(docs[i], top_n=topn)
            # print(similar_topics)
            content_topics.append(similar_topics)
        columns = [f'topic{i}' for i in range(topn)]
        pd.DataFrame(content_topics, columns=columns).to_csv(self.content_topic_file, index=None)


if __name__ == "__main__":
    renminribao = '/home/disk/disk2/lw/covid-19-weibo-processed/renminribao'

    extrator = TopicExtractor(renminribao)
    extrator.extract(topic_num=9, random_seed=24)

