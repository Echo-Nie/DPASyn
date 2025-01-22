# Reproduction-of-the-AttenSyn-paper
AttenSyn: An Attention-Based Deep Graph Neural Network forAnticancer Synergistic Drug Combination Prediction

在对比GCN-三头注意力机制和GAT-DualAttention模型的性能时，我们可以从多个指标进行分析。以下是对每个指标的详细对比和解释：

### 1. **AUC_dev (Area Under Curve - Development)**
- **GCN-三头注意力机制**: 从0.747逐步提升到0.918。
- **GAT-DualAttention**: 从0.757逐步提升到0.923。

**分析**: AUC_dev反映了模型在开发集上的分类能力，可能是因为GAT-Dual结合了图注意力机制和双重注意力机制，能够更好地捕捉节点之间的关系和特征的重要性。

### 2. **PR_AUC (Precision-Recall Area Under Curve)**
- **GCN-三头注意力机制**: 从0.743逐步提升到0.911。
- **GAT-DualAttention**: 从0.748逐步提升到0.915。

**分析**: 结果差不多。

### 3. **ACC (Accuracy)**
- **GCN-三头注意力机制**: 从0.667逐步提升到0.843。
- **GAT-DualAttention**: 从0.682逐步提升到0.862。

**分析**: 准确率反映了模型在所有样本中正确分类的比例，GAT-Dual整体分类性能更好。

### 4. **BACC (Balanced Accuracy)**
- **GCN-三头注意力机制**: 从0.671逐步提升到0.842。
- **GAT-DualAttention**: 从0.671逐步提升到0.861。

**分析**: 平衡准确率考虑了类别不平衡问题，GAT-Dual在处理不平衡数据时更为有效。

### 5. **PREC (Precision)**
- **GCN-三头注意力机制**: 从0.623逐步提升到0.853。
- **GAT-DualAttention**: 从0.773逐步提升到0.853。

**分析**: 但GAT-DualAttention在早期阶段的精确率更高。精确率反映了模型在预测为正类的样本中实际为正类的比例，
GAT-DualAttention的较高精确率表明其在减少假阳性方面更为有效。

### 6. **TPR (True Positive Rate, Recall)**
- **GCN-三头注意力机制**: 从0.748逐步提升到0.856。
- **GAT-DualAttention**: 从0.463逐步提升到0.854。

**分析**: TPR反映了模型在正类样本中正确识别的比例，GCN-三头注意力机制的较高TPR表明其在捕捉正类样本方面更为敏感。

### 7. **KAPPA **
- **GCN-三头注意力机制**: 从0.339逐步提升到0.684。
- **GAT-DualAttention**: 从0.349逐步提升到0.723。
- 提升了**4个点**

**分析**: Kappa系数考虑了随机一致性的影响，GAT-DualAttention的较高Kappa系数表明其分类结果与真实标签的一致性更好。

### 8. **Recall**
- **GCN-三头注意力机制**: 从0.748逐步提升到0.856。
- **GAT-DualAttention**: 从0.463逐步提升到0.854。

**分析**: 差不多，初期GAT-Dual更差

### 9. **F1 Score**
- **GCN-三头注意力机制**: 从0.680逐步提升到0.831。
- **GAT-DualAttention**: 从0.579逐步提升到0.854。
- 最终提升**2个点**

**分析**:F1 Score是精确率和召回率的调和平均数，表明GAT-Dual其在平衡精确率和召回率方面更为有效。
