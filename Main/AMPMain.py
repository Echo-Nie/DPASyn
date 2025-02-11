import random
import time

from torch import amp
from NewTechnology.SynPredTest.utils import *
from model import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score, f1_score
from sklearn import metrics
from torch.utils.data import DataLoader
# 设置随机种子，确保实验可重复
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, device, loader_train, optimizer, epoch, scaler):
    """训练函数"""
    print('Training on {} samples...'.format(len(loader_train.dataset)))
    model.train()  # 设置模型为训练模式
    for batch_idx, (data1, data2, y) in enumerate(loader_train):
        # 将数据移动到设备（GPU或CPU）
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = y.to(device)
        optimizer.zero_grad()  # 梯度清零

        # 使用自动混合精度（AMP）加速训练
        with torch.amp.autocast("cuda", enabled=True):
            output = model(data1, data2)  # 前向传播
            loss = loss_fn(output, y)  # 计算损失

        # 反向传播和参数更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 每隔一定步数打印训练信息
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1.x), len(loader_train.dataset),
                       100. * batch_idx / len(loader_train), loss.item()))


def predicting(model, device, loader_test):
    """预测函数"""
    model.eval()  # 设置模型为评估模式
    total_preds = torch.Tensor()  # 存储所有预测得分
    total_labels = torch.Tensor()  # 存储所有真实标签
    total_prelabels = torch.Tensor()  # 存储所有预测标签
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():  # 禁用梯度计算
        for data1, data2, y in loader_test:
            # 将数据移动到设备（GPU或CPU）
            data1 = data1.to(device)
            data2 = data2.to(device)

            # 使用自动混合精度（AMP）加速预测
            with torch.amp.autocast('cuda', enabled=True):
                output = model(data1, data2)  # 前向传播

            # 对输出进行softmax处理，得到预测概率
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            # 获取预测标签和预测得分
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            # 将结果拼接起来
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1)), 0)
    # 返回真实标签、预测得分和预测标签
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


# 定义模型
modeling = AttenSyn
print(modeling.__name__)

# 超参数设置
TRAIN_BATCH_SIZE = 256  # 训练批次大小
TEST_BATCH_SIZE = 256  # 测试批次大小
LR = 0.0005  # 学习率
LOG_INTERVAL = 20  # 日志打印间隔
NUM_EPOCHS = 120  # 训练轮数

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# 设置设备（GPU或CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = MyDataset()

# 计算数据集长度和划分比例
lenth = len(dataset)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

# 随机打乱数据集
random_num = random.sample(range(0, lenth), lenth)

# 5折交叉验证
for i in range(5):
    # 划分训练集和测试集
    test_num = random_num[pot * i:pot * (i + 1)]
    train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

    # 获取训练数据和测试数据
    data_train = dataset.get_data(train_num)
    data_test = dataset.get_data(test_num)
    # 创建数据加载器
    loader_train = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

    # 初始化模型、损失函数和优化器
    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 定义结果文件路径
    file_result = '../result/DualGAT-AMP-lr{:0.4f}'.format(LR) + str(i) + '.csv'
    # 写入表头
    AUCs = ('Epoch,ACC,PR_AUC,AUC,BACC,PREC,TPR,KAPPA,RECALL,Precision,F1')
    with open(file_result, 'w') as f:
        f.write(AUCs + '\n')

    best_acc = 0  # 记录最佳AUC值
    scaler = torch.amp.GradScaler(enabled=amp)

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()  # 记录每个 epoch 的开始时间

        # 训练模型
        train(model, device, loader_train, optimizer, epoch + 1, scaler)

        # 在测试集上进行预测
        T, S, Y = predicting(model, device, loader_test)

        # 计算性能指标
        AUC = roc_auc_score(T, S)  # 计算AUC
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)  # 计算PR-AUC
        BACC = balanced_accuracy_score(T, Y)  # 计算平衡准确率
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()  # 计算混淆矩阵
        TPR = tp / (tp + fn)  # 计算真正率
        PREC = precision_score(T, Y)  # 计算精确率
        ACC = accuracy_score(T, Y)  # 计算准确率
        KAPPA = cohen_kappa_score(T, Y)  # 计算Kappa系数
        recall = recall_score(T, Y)  # 计算召回率
        precision = precision_score(T, Y)  # 计算精确率
        F1 = f1_score(T, Y)  # 计算F1分数
        AUCs = [epoch, ACC, PR_AUC, AUC, BACC, PREC, TPR, KAPPA, recall, precision, F1]

        # 保存数据
        if best_acc < ACC:
            best_acc = ACC  # 更新最佳ACC值
            save_AUCs(AUCs, file_result)  # 保存结果到CSV文件

        # 计算每个 epoch 的结束时间
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        # 打印每个 epoch 的运行时间
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.')

        # 打印最佳AUC值
        print('best_acc', best_acc)