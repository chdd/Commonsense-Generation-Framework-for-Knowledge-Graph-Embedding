from sklearn.cluster import AgglomerativeClustering
import numpy as np
def aggregate_clusters(cluster_dict):
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
            aggregated_dict[key] = keys

    return aggregated_dict

# 示例用法
cluster_dict = {
    1: [1],
    2: [1],
    3: [2],
    4: [2],
    5: [3]
}

aggregated_dict = aggregate_clusters(cluster_dict)
print(aggregated_dict)