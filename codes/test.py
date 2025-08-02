def similarity(set_A, set_B):
    intersection = len(set_A & set_B)
    magnitude_A = len(set_A)
    magnitude_B = len(set_B)

    # 计算相似度的变体
    similarity_score = 1 - intersection / (magnitude_A + magnitude_B)**2

    return similarity_score

# 示例用法
set_rel2t = {1, 2, 3}
set_tail2conc = {3, 4, 5}

similarity_score = similarity(set_rel2t, set_tail2conc)
print("相似度:", similarity_score)