import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import accuracy_score

# 假设这些模块已经定义
from NewTechnology.SynPredTest.utils.util import MyDataset, collate
from model import AttenSyn

# 设置随机种子
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 超参数搜索空间
HYPERPARAM_SPACE = {
    'learning_rate': [0.001, 0.0005, 0.0001, 0.00005],
    'batch_size': [128, 256, 512],
    'num_epochs': [80, 90, 100, 120],
    'dropout_rate': [0.1, 0.3, 0.5]
}


# 随机搜索参数生成器
def generate_random_params():
    params = {
        'lr': random.choice(HYPERPARAM_SPACE['learning_rate']),
        'batch_size': random.choice(HYPERPARAM_SPACE['batch_size']),
        'num_epochs': random.choice(HYPERPARAM_SPACE['num_epochs']),
        'dropout': random.choice(HYPERPARAM_SPACE['dropout_rate'])
    }
    return params


# 训练函数
def train(model, device, loader_train, optimizer, scaler):
    model.train()
    for data1, data2, y in loader_train:
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            output = model(data1, data2)
            loss = nn.CrossEntropyLoss()(output, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


# 验证函数
def validate(model, device, loader_test):
    model.eval()
    total_preds, total_labels = [], []
    with torch.no_grad():
        for data1, data2, y in loader_test:
            data1, data2 = data1.to(device), data2.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                output = model(data1, data2)
            preds = torch.softmax(output, 1).cpu().numpy()
            total_preds.extend(preds[:, 1])
            total_labels.extend(y.cpu().numpy())
    return np.array(total_labels), np.array(total_preds)


# 训练验证流程
def train_and_validate(params, device, dataset, num_folds=5):
    all_fold_results = []
    lenth = len(dataset)
    pot = int(lenth / num_folds)

    for fold in range(num_folds):
        # 数据划分
        indices = list(range(lenth))
        test_indices = indices[fold * pot: (fold + 1) * pot]
        train_indices = [i for i in indices if i not in test_indices]

        # 创建数据加载器
        train_data = dataset.get_data(train_indices)
        test_data = dataset.get_data(test_indices)
        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=False, collate_fn=collate)

        # 初始化模型和优化器
        model = AttenSyn(dropout_rate=params['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        scaler = torch.amp.GradScaler(enabled=True)

        # 训练和验证
        best_acc = 0
        for epoch in range(params['num_epochs']):
            train(model, device, train_loader, optimizer, scaler)
            labels, preds = validate(model, device, test_loader)
            pred_labels = (preds > 0.5).astype(int)
            current_acc = accuracy_score(labels, pred_labels)
            if current_acc > best_acc:
                best_acc = current_acc

        all_fold_results.append(best_acc)

    return np.mean(all_fold_results)


# 主程序
if __name__ == "__main__":
    # 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = MyDataset()

    # 超参数搜索
    best_params = None
    best_score = 0
    search_records = []

    # 进行50次随机搜索
    for search_iter in range(50):
        print(f"\n=== Hyperparameter Search Iteration {search_iter + 1}/50 ===")

        # 生成随机参数
        current_params = generate_random_params()
        print("Current Parameters:", current_params)

        # 训练验证流程
        start_time = time.time()
        avg_acc = train_and_validate(current_params, device, dataset)
        duration = time.time() - start_time

        # 记录结果
        search_records.append({
            'params': current_params,
            'score': avg_acc,
            'time': duration
        })

        # 更新最佳参数
        if avg_acc > best_score:
            best_score = avg_acc
            best_params = current_params
            print(f"New Best Parameters Found! Accuracy: {avg_acc:.4f}")

        # 打印当前最佳
        print(f"Current Best Accuracy: {best_score:.4f}")
        print(f"Current Best Parameters: {best_params}")

    # 最终输出
    print("\n=== Hyperparameter Search Completed ===")
    print(f"Best Validation Accuracy: {best_score:.4f}")
    print("Best Parameter Combination:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # 保存搜索记录
    with open('hyperparam_search.log', 'w') as f:
        for record in search_records:
            f.write(f"Score: {record['score']:.4f} | Params: {record['params']} | Time: {record['time']:.1f}s\n")