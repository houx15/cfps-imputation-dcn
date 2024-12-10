# CFPS-IMPUTATION-DCN

五种“缺失”模式，所谓缺失，即一部分数据不出现在train set中，而是被放在test set中，所以重点在于train-test的分割

1. 完全随机
2. 随机选择某几个variable某一个或多个year全部消失
3. 随机选择某几个variable全部year全部消失
4. 根据variable的embedding做聚类，选择距离相近的一组variable全部消失
