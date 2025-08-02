# import matplotlib.pyplot as plt
#
# # 读取 entity_dict 文件并将实体存储到字典中
# entity_dict = {}
# with open('D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\\entities.dict', 'r') as file:
#     for line in file:
#         index, entity = line.strip().split('\t')
#         entity_dict[entity] = 0
#
# # 读取 train_txt 文件并统计每个实体的出现次数
# with open('D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\\train.txt', 'r') as file:
#     for line in file:
#         head_entity, relation, tail_entity = line.strip().split('\t')
#         if head_entity in entity_dict:
#             entity_dict[head_entity] += 1
#         if tail_entity in entity_dict:
#             entity_dict[tail_entity] += 1
#
# # 统计不同频率的实体个数
# freq_count = {}
# for freq in entity_dict.values():
#     if freq in freq_count:
#         freq_count[freq] += 1
#     else:
#         freq_count[freq] = 1
#
# # 绘制频率的分布图
# plt.figure(figsize=(10, 6))
# plt.bar( freq_count.values(),freq_count.keys())
# plt.xlabel('Frequency of Entities')
# plt.ylabel('Number of Entities')
# plt.title('Distribution of Entity Frequencies')
# plt.show()
#
import numpy as np
import matplotlib.pyplot as plt

# 读取 entity_dict 文件并将实体存储到字典中
entity_dict = {}
with open('D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\\entities.dict', 'r') as file:
    for line in file:
        index, entity = line.strip().split('\t')
        entity_dict[entity] = 0

# 读取 train_txt 文件并统计每个实体的出现次数
with open('D:\Experiment\work\cge-hake\data_concept\FB15k-237_concept\\train.txt', 'r') as file:
    for line in file:
        head_entity, relation, tail_entity = line.strip().split('\t')
        if head_entity in entity_dict:
            entity_dict[head_entity] += 1
        if tail_entity in entity_dict:
            entity_dict[tail_entity] += 1

# 统计不同频率的实体个数
freq_count = {}
for freq in entity_dict.values():
    if freq in freq_count:
        freq_count[freq] += 1
    else:
        freq_count[freq] = 1

# 按照频率排序
sorted_freq_count = sorted(freq_count.items())

# 获取排序后的频率和实体个数
sorted_freq = [item[0] for item in sorted_freq_count if item[0] != 0]  # 过滤掉横轴为0的点
entity_count = [item[1] for item in sorted_freq_count if item[0] != 0]  # 对应的实体个数

# 对数据进行平滑处理，这里使用简单移动平均
window_size = 10
smoothed_entity_count = np.convolve(entity_count, np.ones(window_size) / window_size, mode='valid')

# 绘制频率的平滑曲线图
plt.figure(figsize=(10, 6),dpi=300)
plt.plot(sorted_freq[window_size - 1:], smoothed_entity_count)
plt.xlabel('Entity ID')
plt.ylabel('Frequency')
# plt.title('Distribution of Entity Frequencies (Smoothed Line Plot)')
plt.grid(True)
plt.savefig('frquent15k.png')
plt.show()



