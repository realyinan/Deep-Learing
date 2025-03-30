import torch
import jieba
import torch.nn as nn

# 一句话
text = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"

# 使用jieba模块进行分词
words = jieba.lcut(text)
print(words)

# 创建词嵌入层
embed = nn.Embedding(num_embeddings=len(words), embedding_dim=8)
for i, word in enumerate(words):
    word_vec = embed(torch.tensor(data=i))
    print(word_vec)
