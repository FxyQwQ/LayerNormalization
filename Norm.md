#### BatchNormalization
- 分别对每个通道中所有样本中的所有特征进行归一化
![alt text](img/image.png)
#### LayerNormalization
- 分别对每个样本中的所有通道/embedding的所有特征/token进行归一化
![alt text](img/image-1.png)
#### InstanceNormalization
- 分别对每个样本的每个通道中的所有特征进行归一化
![alt text](img/image-2.png)
#### GroupNormalization
- 分别对每个样本的每组通道中的所有特征进行归一化
![alt text](img/image-3.png)