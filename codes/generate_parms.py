from matplotlib import pyplot as plt
import networkx as nx
import community  # pip install python-louvain
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def get_rel_dict(rel_path):
    rel_dict = {}
    with open(rel_path, 'r') as file:
        for line in file:
            # 去除行末尾的换行符并使用空格拆分每一行
            id, rel = line.strip().split('\t')
            # 确保每行至少包含三个元素
            rel_dict[rel] = id
    return rel_dict


def get_ent_dict(ent_path):
    ent_dict = {}
    with open(ent_path, 'r', encoding='UTF-8') as file:
        for line in file:
            # 去除行末尾的换行符并使用空格拆分每一行
            id, rel = line.strip().split('\t')
            # 确保每行至少包含三个元素
            ent_dict[rel] = id
    return ent_dict


def get_orgin_entdom(train_triples, rel2dom_h, rel2dom_t, rel_dict, ent_dict):
    dom_ent = {}
    ent_dom = {}
    for h, r, t in train_triples:
        if ent_dict[h] not in ent_dom:
            ent_dom[ent_dict[h]] = []
        ent_dom[ent_dict[h]].append(int(rel2dom_h[rel_dict[r]][0]))
        if ent_dict[t] not in ent_dom:
            ent_dom[ent_dict[t]] = []
        ent_dom[ent_dict[t]].append(int(rel2dom_t[rel_dict[r]][0]))
        # 领域对应的关系
        if rel2dom_h[rel_dict[r]][0] not in dom_ent:
            dom_ent[rel2dom_h[rel_dict[r]][0]] = []
        dom_ent[rel2dom_h[rel_dict[r]][0]].append(int(ent_dict[h]))
        if rel2dom_t[rel_dict[r]][0] not in dom_ent:
            dom_ent[rel2dom_t[rel_dict[r]][0]] = []
        # print(rel2dom_t[rel_dict[r]][0])
        dom_ent[rel2dom_t[rel_dict[r]][0]].append(int(ent_dict[t]))
    return dom_ent, ent_dom


def _get_rel_type(train_triples):
    count_r = {}
    count_h = {}
    count_t = {}
    # for triples in train_triples:
    for h, r, t in train_triples:
        if r not in count_r:
            count_r[r] = 0
            count_h[r] = set()
            count_t[r] = set()
        count_r[r] += 1
        count_h[r].add(h)
        count_t[r].add(t)
    r_tp = {}
    for r in count_r.keys():
        tph = count_r[r] / len(count_h[r])
        hpt = count_r[r] / len(count_t[r])
        if hpt < 1.5:
            if tph < 1.5:
                r_tp[r] = 0  # 1-1
            else:
                r_tp[r] = 1  # 1-M
        else:
            if tph < 1.5:
                r_tp[r] = 2  # M-1
            else:
                r_tp[r] = 3  # M-M
    return r_tp


def quchong(dom_ent, ent_dom):
    dom_ent = {key: list(set(values)) for key, values in dom_ent.items()}
    ent_dom = {key: list(set(values)) for key, values in ent_dom.items()}
    return dom_ent, ent_dom


def get_orgin_rel2dom(rel_dict):
    rel2dom_h = {}
    rel2dom_t = {}
    for rel, id in rel_dict.items():
        rel2dom_h[id] = [int(id)]
        rel2dom_t[id] = [int(int(id) + len(rel_dict))]
    return rel2dom_h, rel2dom_t


# 读取 .dict 文件
train_triple = []
train_txt = r"D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\train.txt"

with open(train_txt, 'r', encoding='UTF-8') as file:
    for line in file:
        # 去除行末尾的换行符并使用空格拆分每一行
        elements = line.strip().split('\t')
        # 确保每行至少包含三个元素
        if len(elements) >= 3:
            train_triple.append([elements[0], elements[1], elements[2]])
    r_tp = _get_rel_type(train_triple)
    file_path = 'my_dict.json'
    print(f"The dictionary has been saved to {file_path}.")
    print(r_tp)
# 首先创建一个字典，无论是什么关系我们都将其置为1
rel_path = r"D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\relation.dict"
ent_path = r"D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\entities.dict"
rel_dict = get_rel_dict(rel_path)

ent_dict = get_ent_dict(ent_path)
rel2dom_h, rel2dom_t = get_orgin_rel2dom(rel_dict)
dom_ent, ent_dom = get_orgin_entdom(train_triple, rel2dom_h, rel2dom_t, rel_dict, ent_dict)
dom_ent, ent_dom = quchong(dom_ent, ent_dom)
rel_num = len(rel_dict)
dict_236 = {key: value for key, value in dom_ent.items() if key <= rel_num - 1}
dict_237 = {key: value for key, value in dom_ent.items() if key >= rel_num}


def generateFeature(dict_236, ent_dict):
    feature_236 = {}
    for key, value in dict_236.items():
        feature = np.zeros(len(ent_dict))
        for i in range(len(ent_dict)):
            if i not in value:
                feature[i] = 0
            else:
                feature[i] = 1
        feature_236[key] = feature
    return feature_236


feature_236 = generateFeature(dict_236, ent_dict)
feature_237 = generateFeature(dict_237, ent_dict)
# 假设 features_embedded 是你的数据
features = np.array(list(feature_236.values()))
tsne = TSNE(n_components=2,
            perplexity=30,  # 增大困惑度
            learning_rate=100,
            random_state=42)
features_embedded = tsne.fit_transform(features)
# 尝试不同的 eps 和 min_samples 组合
eps_values = [0.5, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.3, 3.4, 3.5]
min_samples_values = [2, 3, 4, 5, 6]

best_eps = None
best_min_samples = None
best_num_clusters = 0
best_noise_points = float('inf')

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_embedded)
        # print(f'features_embedded:{features_embedded}')
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # 排除噪声点
        noise_points = np.sum(cluster_labels == -1)
        print(f"eps: {eps}, min_samples: {min_samples}, num_clusters: {num_clusters}, noise_points: {noise_points}")

        # 找到噪声点少的组合
        if noise_points < best_noise_points or (noise_points == best_noise_points and num_clusters < best_num_clusters):
            best_eps = eps
            best_min_samples = min_samples
            best_num_clusters = num_clusters
            best_noise_points = noise_points

print(
    f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best num_clusters: {best_num_clusters}, Best noise_points: {best_noise_points}")

# 使用最佳参数进行聚类
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
cluster_labels = dbscan.fit_predict(features_embedded)

# 统计每个聚类的数量
cluster_counts = {}
for label in set(cluster_labels):
    cluster_counts[label] = np.sum(cluster_labels == label)

# 输出每个聚类的数量
for label, count in cluster_counts.items():
    if label == -1:
        print(f"噪声点数量: {count}")
    else:
        print(f"聚类 {label} 的数量: {count}")

# 可视化聚类结果
plt.figure(figsize=(8, 6), dpi=500)
for label in np.unique(cluster_labels):
    if label == -1:  # Label -1 represents noise points
        plt.scatter(features_embedded[cluster_labels == label][:, 0],
                    features_embedded[cluster_labels == label][:, 1],
                    color='black')
    else:
        plt.scatter(features_embedded[cluster_labels == label][:, 0],
                    features_embedded[cluster_labels == label][:, 1])

plt.title("DBSCAN Clustering Result")
plt.legend()
plt.show()
