from matplotlib import pyplot as plt
import networkx as nx
import community  # pip install python-louvain
from sklearn.decomposition import PCA

rel_type = {}


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


def get_orgin_rel2dom(rel_dict):
    rel2dom_h = {}
    rel2dom_t = {}
    for rel, id in rel_dict.items():
        rel2dom_h[id] = [int(id)]
        rel2dom_t[id] = [int(int(id) + len(rel_dict))]
    return rel2dom_h, rel2dom_t


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


def quchong(dom_ent, ent_dom):
    dom_ent = {key: list(set(values)) for key, values in dom_ent.items()}
    ent_dom = {key: list(set(values)) for key, values in ent_dom.items()}
    return dom_ent, ent_dom


from collections import defaultdict


def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def aggregate_keys_by_similarity(original_dict, threshold):
    result_list = []

    keys = list(original_dict.keys())
    visited = set()

    for i, key1 in enumerate(keys):
        if key1 in visited:
            continue

        visited.add(key1)
        aggregated_keys = [key1]

        for j, key2 in enumerate(keys[i + 1:]):
            j += i + 1
            if key2 not in visited:
                similarity = jaccard_similarity(original_dict[key1], original_dict[key2])
                if similarity >= threshold:
                    aggregated_keys.append(key2)
                    visited.add(key2)

        result_list.append(aggregated_keys)

    return result_list


from sklearn.cluster import DBSCAN
import numpy as np
# 定义 DBSCAN 聚类函数
from sklearn.cluster import AgglomerativeClustering


def aggregate_clusters(cluster_dict, rel_num):
    value_to_keys = {}  # 用于记录相同值对应的原始键
    for key, value_list in cluster_dict.items():
        for value in value_list:
            if value in value_to_keys:
                value_to_keys[value].append(key)
            else:
                value_to_keys[value] = [key]

    # 构建聚合后的字典
    aggregated_dict = {}
    for value, keys in value_to_keys.items():
        for key in keys:
            new_key = key - rel_num if key >= rel_num else key
            aggregated_dict[new_key] = keys

    return aggregated_dict


from sklearn.manifold import TSNE
from collections import Counter
import re


def load_mid2name(filepath):
    """加载 Freebase MID 到实体名称的映射"""
    mid2name = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mid, name = parts
                    mid2name[mid] = name
    return mid2name


mid2name = load_mid2name(r"D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\FB15k_mid2name.txt")
# 全局概念映射规则（按优先级排序）
concept_rules = [
    (r"Ohio_State_University", "University"),
    (r"University_of_Miami", "University"),
    (r"University_of_Richmond", "University"),
    (r"United_States_of_America", "Country"),
    (r"Canada", "Country"),
    (r"France", "Country"),
    (r"King_Kong", "Film"),
    (r"Dangerous_Liaisons", "Film"),
    (r"A_Midsummer_Night's_Dream", "Film"),
    (r"The_Dark_Knight_Rises", "Film"),
    (r"X-Men_Origins:_Wolverine", "Film"),
    (r"Hellboy_II:_The_Golden_Army", "Film"),
    (r"Spike_Jonze", "Film professionals"),
    (r"J.J._Abrams", "Film professionals"),
    (r"Irrfan_Khan", "Film professionals"),
    (r"United_States_Dollar", "Economy"),
    (r"Euro", "Economy"),
    (r"Sampler", "Music Tool"),
    (r"Percussion", "Music Tool"),
    (r"Drum", "Music Tool"),
    (r"Carson_Daly", "Performer"),
    (r"Ben_Affleck", "Performer"),
    (r"Kate_Beckinsale", "Performer"),
    (r"Michael_Jackson", "Celebrities"),
    (r"Madonna", "Celebrities"),
    (r"Steven_Spielberg", "Celebrities"),
    (r"Luxembourg", "Country"),
    (r"Uruguay", "Country"),
    (r"Mexico", "Country"),
    (r".", "Others")
]


def name_to_concept(entity_name):
    """将实体名称映射到概念类别"""
    for pattern, concept in concept_rules:
        if re.search(pattern, entity_name, flags=re.IGNORECASE):
            return concept
    return "others"
manual_colors = [
    '#FF0000', '#00FF00', '#0000FF',  # 红、绿、蓝
    '#FFD700', '#FF00FF', '#00FFFF',  # 金、粉、青
    '#FFA500', '#800080', '#008000'   # 橙、紫、深绿
]

def dbscan_clustering(original_dict, raw_dict, eps, min_samples, top_k,pca_dim):
    features = np.array(list(original_dict.values()))
    pca = PCA(n_components=pca_dim)
    features_pca = pca.fit_transform(features)
    # t-SNE降维优化
    tsne = TSNE(n_components=2,
                perplexity=30,
                learning_rate=100,
                random_state=42)
    features_embedded = tsne.fit_transform(features_pca)
    # DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features_embedded)
    # 聚类簇数和标签分析
    label_counter = Counter(cluster_labels)
    # 选出top_k个最大的簇（排除噪声 -1）
    major_clusters = [label for label, count in label_counter.items()
                      if label != -1]
    major_clusters = sorted(major_clusters, key=lambda l: label_counter[l], reverse=True)[:top_k]
    keys_list = list(original_dict.keys())  # 转换为列表
    # 创建画布
    plt.figure(figsize=(8, 6), dpi=500)

    # 颜色映射
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = manual_colors[:n_clusters]


    # 绘制噪声点
    noise_mask = (cluster_labels == -1)
    if np.any(noise_mask):
        plt.scatter(features_embedded[noise_mask][:, 0],
                    features_embedded[noise_mask][:, 1],
                    c='#999999', s=10, alpha=0.1,
                    label=f'Noise')

    for idx, label in enumerate(major_clusters):
        cluster_mask = (cluster_labels == label)
        entity_names = []
        for key in np.array(keys_list)[cluster_mask]:
            entity_ids = raw_dict[key]
            names = [id_to_ent.get(str(id), f"Unknown_{id}") for id in entity_ids]
            entity_names.extend(names)
        # if entity_names:
        #     top_names = Counter(entity_names).most_common(10)
        #     for entity, count in top_names:
        #         name = mid2name.get(entity)
        #         print(f"name:{name},出现次数: {count}")
        # else:
        #     top_names = "Unnamed"
        # 取出现最多的前三个实体 MID
        # top3 = [mid for mid, _ in Counter(entity_names).most_common(3)]
        # # 映射到名称
        # top3_names = [mid2name.get(mid, mid) for mid in top3]
        # # 将名称映射到概念
        # concepts = [name_to_concept(name) for name in top3_names]
        # # 统计最常见概念
        # concept_label = Counter(concepts).most_common(1)[0][0] if concepts else "Unnamed"
        concept_label = f"Conquer{idx + 1}"
        print(concept_label)
        plt.scatter(features_embedded[cluster_mask][:, 0],
                    features_embedded[cluster_mask][:, 1],
                    c=[colors[idx]], s=60, alpha=0.9,
                    label=concept_label)
    # 图例优化
    plt.legend(loc='upper left',
               frameon=True, )
    plt.show()

    # 返回聚类字典
    cluster_dict = {}
    for i, key in enumerate(original_dict.keys()):
        cluster_dict[key] = [cluster_labels[i]]
    return cluster_dict


# Convert the clustered points to a dictionary where the keys are cluster IDs
def convert_to_cluster_dict(cluster_labels, original_dict):
    cluster_dict = {}
    for i, key in enumerate(original_dict.keys()):
        cluster_dict[key] = cluster_labels[i]
    return cluster_dict


# 使用 Jaccard 相似度阈值为 0.5 进行聚合

def get_final_rel2dom_clusting(rel2dom_h, rel2dom_t, result_236, result_237, rel_num):
    rel2dom_h_final = {}
    rel2dom_t_final = {}
    for i in range(0, rel_num):
        for son_list in result_236:
            if i in son_list:
                rel2dom_h_final[i] = son_list
            else:
                continue
        for son_t_list in result_237:
            if i + rel_num in son_t_list:
                rel2dom_t_final[i] = son_t_list
            else:
                continue
    return rel2dom_h_final, rel2dom_t_final


# 聚合相似实体观念
def get_final_rel2dom(rel2dom_h, rel2dom_t, result_236, result_237, rel_num):
    rel2dom_h_final = {}
    rel2dom_t_final = {}
    for i in range(0, rel_num):
        for son_list in result_236:
            if i in son_list:
                rel2dom_h_final[i] = son_list
            else:
                continue
        for son_t_list in result_237:
            if i + rel_num in son_t_list:
                rel2dom_t_final[i] = son_t_list
            else:
                continue
    return rel2dom_h_final, rel2dom_t_final


# 尾实体：72，104，  #73，171  #206,33,234 #[310, 408] "128": [345, 365]
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


import json

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

id_to_ent = {str(v): k for k, v in ent_dict.items()}
rel2dom_h, rel2dom_t = get_orgin_rel2dom(rel_dict)
dom_ent, ent_dom = get_orgin_entdom(train_triple, rel2dom_h, rel2dom_t, rel_dict, ent_dict)
dom_ent, ent_dom = quchong(dom_ent, ent_dom)
# print("dom_ent:",dom_ent)
# print(rel2dom_h,rel2dom_t)

# 将键分为小于等于 236 和大于等于 237 两种情况
rel_num = len(rel_dict)
dict_236 = {key: value for key, value in dom_ent.items() if key <= rel_num - 1}
dict_237 = {key: value for key, value in dom_ent.items() if key >= rel_num}

# 使用 Jaccard 相似度阈值为 0.5 进行聚合
threshold = 0.9
result_236 = aggregate_keys_by_similarity(dict_236, threshold)
result_237 = aggregate_keys_by_similarity(dict_237, threshold)
print("Cluster for keys <= rel_num:", result_236)
print("Cluster for keys >= rel_num:", result_237)

# 进行最后一步，将具有相似实体的概念分发到各个地方
rel2dom_h_final, rel2dom_t_final = get_final_rel2dom(rel2dom_h, rel2dom_t, result_236, result_237, rel_num)
print("rel2dom_h_final=", rel2dom_h_final)
print(rel2dom_t_final)
rel2dom_h_final_path = 'rel2dom_h.json'
rel2dom_t_final_path = 'rel2dom_t.json'
dom_ent_path = 'dom_ent.json'
ent_dom_path = 'ent_dom.json'


# with open(rel2dom_h_final_path,'w')as file :
#     json.dump(rel2dom_h_final,file)
#     file.close()
# with open(rel2dom_t_final_path,'w')as file :
#     json.dump(rel2dom_t_final,file)
#     file.close()
# with open(dom_ent_path,'w')as file :
#     json.dump(dom_ent,file)
#     file.close()
# with open(ent_dom_path,'w')as file :
#     json.dump(ent_dom,file)
#     file.close()
# rel2nn={}
# for key,value in r_tp.items():
#     rel2nn[rel_dict[key]]=value
# print(rel2nn)
# rel2nn_path="rel2nn.json"
# with open(rel2nn_path,'w')as file :
#     json.dump(rel2nn,file)
#     file.close()
# 对于n-n关系来说，n-n的关系都是同属于一个概念，即n-n关系头实体尾实体都是一个群体， concept_h n-n concept_t
# 对于1-1关系来说，也是一样的
# 对于n-1关系来说，
# 应该换个角度，每个实体链接一个关系就应该有不同的concept
# 先搞rel2dom_h，每个三元组的头实体会

# 在所有实体拥有对应的dom后，再进行探查，看哪两个dom出现的频率比较高，再插入到rel——dom那个字典里面，注意，关系头对应的dom只能和关系头对应的dom，二者不可以混为一谈。


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
#
#
# result_236 = dbscan_clustering(feature_236, 2.5, 3)
# result_237 = dbscan_clustering(feature_237, 2, 3)
# result_236 = dbscan_clustering(feature_236, dict_236, 1.5, 3, 5)
# result_237 = dbscan_clustering(feature_237, dict_237, 1.2, 3, 5)
result_236 = dbscan_clustering(feature_236, dict_236, 1.5, 3, 6,30)
result_237 = dbscan_clustering(feature_237, dict_237, 1.5, 3, 6,50)
rel2dom_h_final = aggregate_clusters(result_236, rel_num)
rel2dom_t_final = aggregate_clusters(result_237, rel_num)
print("Cluster for keys <= rel_num using DBSCAN:", result_236)
print("Cluster for keys >= rel_num using DBSCAN:", result_237)
# 这行已经不需要了，因为已经成为了我要的东西
# rel2dom_h_final,rel2dom_t_final=get_final_rel2dom_clusting(rel2dom_h,rel2dom_t,result_236,result_237,rel_num)
print(rel2dom_h_final)
print(rel2dom_t_final)
rel2dom_h_final_path = 'rel2dom_h.json'
rel2dom_t_final_path = 'rel2dom_t.json'
dom_ent_path = 'dom_ent.json'
ent_dom_path = 'ent_dom.json'
with open(rel2dom_h_final_path, 'w') as file:
    json.dump(rel2dom_h_final, file)
    file.close()
with open(rel2dom_t_final_path, 'w') as file:
    json.dump(rel2dom_t_final, file)
    file.close()
with open(dom_ent_path, 'w') as file:
    json.dump(dom_ent, file)
    file.close()
with open(ent_dom_path, 'w') as file:
    json.dump(ent_dom, file)
    file.close()
rel2nn = {}
for key, value in r_tp.items():
    rel2nn[rel_dict[key]] = value
# print(rel2nn)
rel2nn_path = "rel2nn.json"
with open(rel2nn_path, 'w') as file:
    json.dump(rel2nn, file)
    file.close()

