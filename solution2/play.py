import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import glob, os
from utils import vggish_melspectrogram
import pandas as pd
import numpy as np
import random

torch.manual_seed(1)
# 设置超参数

epoches = 2
batch_size = 64
time_step = 28
input_size = 28
learning_rate = 0.01
hidden_size = 64
num_layers = 1

# 测试数据集
all_audios = os.listdir("../data/train/")
test_rate = 0.05
test_num = int(len(all_audios) * test_rate)
test_audios = random.sample(all_audios, test_num)


def test_data(test_audios, num_class=24):
    root = "../data/train/"
    test_x = np.ndarray(shape=(len(test_audios), 5998, 128))
    test_y = np.ndarray(shape=(len(test_audios), num_class))

    data_info = pd.read_csv("../data/train_all.csv")
    gs = data_info.groupby("recording_id")

    for idx, audio in enumerate(test_audios):
        mel = vggish_melspectrogram(os.path.join(root, audio))
        test_x[idx, :, :] = mel
        g = gs.get_group(audio[:-5])
        label = np.zeros(num_class)
        for ridx in range(len(g)):
            row = g.iloc[ridx]
            sp_id = row["species_id"]
            if row["positive"] == 1:
                label[sp_id] = 1
            # else:
            #     label[sp_id] = -1
        test_y[idx] = label

    return test_x, test_y


# LRAP. Instance-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = (scores.sum(-1) / labels.sum(-1)).mean()
    return score.item()

class LRAP_loss(nn.Module):
    def __init__(self):
        super(LRAP_loss, self).__init__()

    def forward(self, preds, labels):
        ranked_classes = torch.argsort(preds, dim=-1, descending=True)
        # i, j corresponds to rank of prediction in row i
        class_ranks = torch.zeros_like(ranked_classes)
        for i in range(ranked_classes.size(0)):
            for j in range(ranked_classes.size(1)):
                class_ranks[i, ranked_classes[i][j]] = j + 1
        # Mask out to only use the ranks of relevant GT labels
        ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
        # All the GT ranks are in front now
        sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
        pos_matrix = torch.tensor(np.array([i + 1 for i in range(labels.size(-1))])).unsqueeze(0)
        score_matrix = pos_matrix / sorted_ground_truth_ranks
        score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
        scores = score_matrix * score_mask_matrix
        score = (scores.sum(-1) / labels.sum(-1)).mean()

        return score

# label-level average
# Assume float preds [BxC], labels [BxC] of 0 or 1
def LWLRAP(preds, labels):
    # Ranks of the predictions
    ranked_classes = torch.argsort(preds, dim=-1, descending=True)
    # i, j corresponds to rank of prediction in row i
    class_ranks = torch.zeros_like(ranked_classes)
    for i in range(ranked_classes.size(0)):
        for j in range(ranked_classes.size(1)):
            class_ranks[i, ranked_classes[i][j]] = j + 1
    # Mask out to only use the ranks of relevant GT labels
    ground_truth_ranks = class_ranks * labels + (1e6) * (1 - labels)
    # All the GT ranks are in front now
    sorted_ground_truth_ranks, _ = torch.sort(ground_truth_ranks, dim=-1, descending=False)
    # Number of GT labels per instance
    num_labels = labels.sum(-1)
    pos_matrix = torch.tensor(np.array([i+1 for i in range(labels.size(-1))])).unsqueeze(0)
    score_matrix = pos_matrix / sorted_ground_truth_ranks
    score_mask_matrix, _ = torch.sort(labels, dim=-1, descending=True)
    scores = score_matrix * score_mask_matrix
    score = scores.sum() / labels.sum()
    return score.item()


def test_filter(x)->bool:
    if not x.endswith(".flac"):
        return False
    if x in test_audios:
        return False
    return True


# 自定义数据集
class AudioDataSet(Data.Dataset):
    def __init__(self, root, data_info, num_class=24, repeat=1):
        self.root = root
        self.data_info = pd.read_csv(data_info)
        self.audio_file_list = list(filter(test_filter, os.listdir(root)))  # glob.glob(os.path.join(root, "*.flac"))
        self.num_class = num_class
        self.groups = self.data_info.groupby("recording_id")
        self.repeat = repeat

    def __len__(self):
        # return self.audio_file_list.__len__()
        if self.repeat == 1:
            return 10000000
        else:
            return self.audio_file_list.__len__() * self.repeat

    def __getitem__(self, item):
        # print(item)
        item = item % self.audio_file_list.__len__()
        audio = self.audio_file_list[item]
        data = vggish_melspectrogram(os.path.join(self.root, audio))
        label = np.zeros(self.num_class)
        group = self.groups.get_group(audio[:-5])
        for idx in range(len(group)):
            row = group.iloc[idx]
            sp_id = row["species_id"]
            if row["positive"] == 1:
                label[sp_id] = 1
            # else:
            #     label[sp_id] = -1
        return data, label


def test_AudioDataSet():
    root = "../data/train/"
    data_info = "../data/train_all.csv"
    batch_size = 64
    epoches = 3
    dataset = AudioDataSet(root, data_info, num_class=24, repeat=10)
    loader = Data.DataLoader(dataset=dataset, batch_size=64,shuffle=True, num_workers=0)

    for epoch in range(epoches):
        for idx, (bth_x, bth_y) in enumerate(loader):
            print(f"{idx} / {epoch}")
            print(bth_x.shape)
            print(bth_y)


class LSTM_RNN(nn.Module):
    """搭建LSTM神经网络"""
    def __init__(self, input_size, hidden_size, num_layers, num_class=10, batch_first=True):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,   # rnn 隐藏单元数
                            num_layers=num_layers,     # rnn 层数
                            batch_first=batch_first,  # If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: False
                            )
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=num_class)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = torch.tensor(x, dtype=torch.float32)
        lstm_out, (h_n, h_c) = self.lstm(x, None)   # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        output = self.output_layer(lstm_out[:, -1, :])   # 选择最后时刻lstm的输出
        return output


def main():
    # 训练集
    train_dataset = torchvision.datasets.MNIST(root="./mnist/", train=True, transform=torchvision.transforms.ToTensor())
    # 测试集
    test_dataset = torchvision.datasets.MNIST(root="./mnist/", train=False, transform=torchvision.transforms.ToTensor())
    test_x = test_dataset.test_data.type(torch.FloatTensor)[:2000] / 255
    test_y = test_dataset.test_labels[:2000]

    # print(test_dataset.test_data)
    # print(test_dataset.test_data.size())
    # plt.imshow(test_dataset.test_data[1].numpy())
    # plt.show()

    # 将训练级集入Loader中
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    lstm = LSTM_RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    print(lstm)
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.view(-1, 28, 28)
            output = lstm(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = lstm(test_x)
                pred_y = torch.max(test_output, dim=1)[1].data.numpy()
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = lstm(test_x[:10])
    pred_y = torch.max(test_output, dim=1)[1].data.numpy().squeeze()
    print(pred_y)
    print(test_y[:10])


def train_audio():
    root = "../data/train/"
    data_info = "../data/train_all.csv"
    batch_size = 8
    epoches = 3
    dataset = AudioDataSet(root, data_info, num_class=24, repeat=10)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_x, test_y = test_data(test_audios)

    lstm = LSTM_RNN(input_size=128, hidden_size=64, num_layers=2, num_class=24)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    loss_function = LRAP_loss()  # nn.MultiLabelSoftMarginLoss()  # LRAP

    for epoch in range(epoches):
        for idx, (bth_x, bth_y) in enumerate(loader):
            print(f"{idx} / {epoch}")
            # print(bth_x)
            # print(bth_y)
            output = lstm(bth_x)
            loss = loss_function(output, bth_y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if idx % 20 == 0:
                train_loss = loss.data.numpy()
                test_output = lstm(test_x)
                # pred_y = torch.max(test_output, dim=1)[1].data.numpy()
                loss = loss_function(test_output, torch.tensor(test_y))
                # accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % train_loss, '| test loss: %.2f' % loss.data.numpy())


if __name__ == "__main__":
    # main()
    # test_AudioDataSet()
    train_audio()