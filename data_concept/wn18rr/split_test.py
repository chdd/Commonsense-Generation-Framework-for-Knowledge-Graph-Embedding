# 读取 train.txt 文件并统计每个实体的出现频率
entity_freq = {}
with open('D:\Experiment\work\cge-hake\data_concept\wn18rr\\train.txt', 'r') as file:
    for line in file:
        head_entity, _, tail_entity = line.strip().split('\t')
        # 统计头实体频率
        if head_entity in entity_freq:
            entity_freq[head_entity] += 1
        else:
            entity_freq[head_entity] = 1
        # 统计尾实体频率
        if tail_entity in entity_freq:
            entity_freq[tail_entity] += 1
        else:
            entity_freq[tail_entity] = 1

# 根据出现频率排序实体
sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)

# 确定常见实体和非常见实体的阈值
total_entities = len(sorted_entities)
common_threshold = int(0.2 * total_entities)
rare_threshold = int(0.8 * total_entities)

# 获取常见实体和非常见实体
common_entities = [entity[0] for entity in sorted_entities[:common_threshold]]
rare_entities = [entity[0] for entity in sorted_entities[common_threshold:]]

# 遍历三元组并将其分为常见三元组和非常见三元组
common_triples = []
rare_triples = []
with open('D:\Experiment\work\cge-hake\data_concept\wn18rr\\test.txt', 'r') as file:
    for line in file:
        head_entity, _, tail_entity = line.strip().split('\t')
        # 判断头实体和尾实体类型
        if head_entity in common_entities and tail_entity in common_entities:
            common_triples.append(line)
        elif head_entity in rare_entities or tail_entity in rare_entities:
            rare_triples.append(line)

# 将常见三元组和非常见三元组分别保存到文件中
with open('../wn18rr_split/common_triples.txt', 'w') as file:
    file.writelines(common_triples)

with open('../wn18rr_split/rare_triples.txt', 'w') as file:
    file.writelines(rare_triples)

print("常见三元组已保存到 common_triples.txt 文件中。")
print("非常见三元组已保存到 rare_triples.txt 文件中。")