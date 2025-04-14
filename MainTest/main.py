import random
import time
from model import *
from NewTechnology.SynPredTest.utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
from sklearn import metrics
from torch.utils.data import DataLoader

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    start_memory = torch.cuda.memory_allocated(device)  # 记录训练前的内存使用量
    criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.MSELoss()

    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(data1, data2)

        if isinstance(outputs, tuple):
            main_output, aux_output = outputs
            # 主损失
            main_loss = criterion(main_output, target)
            # 辅助损失
            aux_loss = aux_criterion(aux_output, (target == 1).float())
            # 总损失
            loss = main_loss + 0.1 * aux_loss
        else:
            loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data1)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        end_memory = torch.cuda.memory_allocated(device)  # 记录训练后的内存使用量
        memory_usage_mb = (end_memory - start_memory) / (1024 ** 2)  # 计算内存增量，单位为MB
        print(f'Training memory usage for epoch {epoch}: {memory_usage_mb:.2f} MB')

def predicting(model, device, loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for data1, data2, y in loader_test:
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1)), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

modeling = AttenSyn
print(modeling.__name__)

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 120

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = MyDataset()

lenth = len(dataset)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    test_num = random_num[pot * i:pot * (i + 1)]
    train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

    data_train = dataset.get_data(train_num)
    data_test = dataset.get_data(test_num)
    loader_train = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 每 120 个 epoch 调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.95)

    best_acc = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()  # 记录每个 epoch 的开始时间
        train(model, device, loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, loader_test)

        # 计算性能指标
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)
        precision = precision_score(T, Y)
        F1 = f1_score(T, Y)
        AUCs = [epoch, ACC, PR_AUC, AUC, BACC, PREC, TPR, KAPPA, recall, precision, F1]

        # 每 120 个 epoch 保存一次结果
        if (epoch + 1) % 120 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            file_result = f'./Result/GAT-atten_dropout_0_lr_{current_lr}_fold_{i}.csv'
            save_AUCs(AUCs, file_result)

        # 更新最佳准确率
        if best_acc < ACC:
            best_acc = ACC

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        # 打印每个 epoch 的运行时间
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.')
        # 调整学习率
        scheduler.step()

        print('best_acc：', best_acc)