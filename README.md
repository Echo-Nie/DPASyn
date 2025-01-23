# AttenSyn: An Attention-Based Deep Graph Neural Network forAnticancer Synergistic Drug Combination Prediction

# Data

Download data.pt file from [here](https://drive.google.com/file/d/1Gqt4HxbUVILIbp17L6e_zLGA_3sVKOw1/view?usp=sharing), and put it into data directory  

</br>

# Paper result

![论文结果](Images/论文结果.png)

| Methods     | AUROC       | AUPR        | ACC         | BACC        | PREC        | TPR         | KAPPA       |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| AttenSyn    | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.84 ± 0.01 | 0.84 ± 0.02 | 0.83 ± 0.02 | 0.82 ± 0.03 | 0.67 ± 0.03 |
| DTSyn       | 0.89 ± 0.01 | 0.87 ± 0.01 | 0.81 ± 0.01 | 0.81 ± 0.02 | 0.84 ± 0.02 | 0.74 ± 0.05 | 0.61 ± 0.03 |
| MR-GNN      | 0.90 ± 0.01 | 0.90 ± 0.01 | 0.82 ± 0.01 | 0.82 ± 0.01 | 0.81 ± 0.02 | 0.80 ± 0.03 | 0.65 ± 0.02 |
| DeepSynergy | 0.72 ± 0.01 | 0.77 ± 0.03 | 0.72 ± 0.01 | 0.72 ± 0.01 | 0.73 ± 0.05 | 0.64 ± 0.02 | 0.43 ± 0.02 |
| RF          | 0.74 ± 0.03 | 0.73 ± 0.03 | 0.67 ± 0.01 | 0.67 ± 0.02 | 0.70 ± 0.07 | 0.59 ± 0.03 | 0.35 ± 0.04 |
| Adaboost    | 0.74 ± 0.02 | 0.72 ± 0.03 | 0.75 ± 0.02 | 0.66 ± 0.02 | 0.63 ± 0.08 | 0.69 ± 0.08 | 0.32 ± 0.04 |
| SVM         | 0.68 ± 0.05 | 0.65 ± 0.06 | 0.62 ± 0.05 | 0.62 ± 0.05 | 0.59 ± 0.05 | 0.66 ± 0.06 | 0.25 ± 0.09 |
| MLP         | 0.84 ± 0.01 | 0.82 ± 0.01 | 0.76 ± 0.01 | 0.75 ± 0.01 | 0.75 ± 0.01 | 0.71 ± 0.01 | 0.50 ± 0.02 |
| Elastic net | 0.68 ± 0.08 | 0.67 ± 0.07 | 0.63 ± 0.07 | 0.63 ± 0.07 | 0.61 ± 0.08 | 0.62 ± 0.07 | 0.27 ± 0.14 |

</br>

# My Reproduction

KAPPA提升5个点；ACC提升2个点；BACC提升2个点；PREC提升两个点；TPR提升3个点；F1分数提升2个点。

![resultFinal](Images/resultFinal.png)

|   Methods    |  AUROC   |   AUPR   |   ACC    |   BACC   |   PREC   |   TPR    |  KAPPA   |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
|   AttenSyn   |   0.92   |   0.91   |   0.84   |   0.84   |   0.83   |   0.82   |   0.67   |
| **dual-GAT** | **0.92** | **0.91** | **0.86** | **0.86** | **0.85** | **0.85** | **0.72** |
|    GATGCN    |  0.9220  |  0.9158  |  0.8531  |  0.8533  |  0.8395  |  0.8574  |  0.7059  |
|   dual-GCN   |  0.9182  |  0.9104  |  0.8493  |  0.8483  |  0.8566  |  0.8235  |  0.6977  |
|     GAT      |  0.9173  |  0.9110  |  0.8542  |  0.8536  |  0.8546  |  0.8385  |  0.7077  |

</br>

# Evaluation（AttenSyn and GAT-Dual）

在对比AttenSyn和GAT-DualAttention模型的性能时，从多个指标进行以下分析。

1. **AUC_dev (Area Under Curve - Development)**

   - **AttenSyn**: 从0.747逐步提升到0.918。

   - **GAT-DualAttention**: 从0.757逐步提升到0.923。


&emsp;分析: AUC_dev反映了模型在开发集上的分类能力，可能是因为GAT-Dual结合了图注意力机制和双重注意力机制，能够更好地捕捉节点之间的关系和特征的重要性。

</br>

2. **PR_AUC (Precision-Recall Area Under Curve)**

   - AttenSyn: 从0.743逐步提升到0.911。

   - GAT-DualAttention: 从0.748逐步提升到0.915。


&emsp;分析: 结果差不多。

</br>

3. **ACC (Accuracy)**

   - AttenSyn: 从0.667逐步提升到0.843。

   - GAT-DualAttention: 从0.682逐步提升到0.862。
   - 提升**2个点**


&emsp;分析: 准确率反映了模型在所有样本中正确分类的比例，GAT-Dual整体分类性能更好。

</br>

4. **BACC (Balanced Accuracy)**
- AttenSyn: 从0.671逐步提升到0.842。
  
- GAT-DualAttention: 从0.671逐步提升到0.861。


&emsp;分析: 平衡准确率考虑了类别不平衡问题，GAT-Dual在处理不平衡数据时更为有效。

</br>

5. **PREC (Precision)**

   - AttenSyn: 从0.623逐步提升到0.853。

   - GAT-DualAttention: 从0.773逐步提升到0.853。

&emsp;分析: 但GAT-DualAttention在早期阶段的精确率更高。精确率反映了模型在预测为正类的样本中实际为正类的比例，
GAT-DualAttention的较高精确率表明其在减少假阳性方面更为有效。

</br>

6. **TPR (True Positive Rate, Recall)**

   - AttenSyn: 从0.748逐步提升到0.856。

   - GAT-DualAttention: 从0.463逐步提升到0.854。


&emsp;分析: TPR反映了模型在正类样本中正确识别的比例，GCN-三头注意力机制的较高TPR表明其在捕捉正类样本方面更为敏感。

</br>

7. **KAPPA**

   - AttenSyn: 从0.339逐步提升到0.684。

   - GAT-DualAttention: 从0.349逐步提升到0.723。

   - 提升了**4个点**


&emsp;分析: Kappa系数考虑了随机一致性的影响，GAT-DualAttention的较高Kappa系数表明其分类结果与真实标签的一致性更好。

</br>

8. **Recall**

   - AttenSyn: 从0.748逐步提升到0.856。

   - GAT-DualAttention: 从0.463逐步提升到0.854。


&emsp;分析: 差不多，初期GAT-Dual更差

</br>

9. **F1 Score**

   - AttenSyn: 从0.680逐步提升到0.831。

   - GAT-DualAttention: 从0.579逐步提升到0.854。

   - 最终提升**2个点**


&emsp;分析:F1 Score是精确率和召回率的调和平均数，表明GAT-Dual其在平衡精确率和召回率方面更为有效。
