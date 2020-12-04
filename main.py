from os.path import join
import torch
from torch_geometric.datasets import TUDataset
from dataset import DiagDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
import shutil
from networks import Net
import torch.nn.functional as F
import argparse
from tensorboard_logger import tensorboard_logger
import numpy as np
import os
from torch.utils.data import random_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from utils import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='twitter',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=150,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
# dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
dataset = DiagDataset(root=join(settings.DATA_DIR, args.dataset))

tensorboard_log_dir = 'tensorboard/%s_%s' % ("sagpool", args.dataset)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

args.num_classes = 2
args.num_features = dataset.num_features
print("num features", args.num_features)

num_training = int(len(dataset) * 0.5)
num_val = int(len(dataset) * 0.75) - num_training
num_test = len(dataset) - (num_training + num_val)
# training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
training_set = dataset[:num_training]
validation_set = dataset[num_training:(num_training+num_val)]
test_set = dataset[(num_training+num_val):]

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model, epoch, loader, thr=None, return_best_thr=False):
    model.eval()
    correct = 0.
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []

    for data in loader:
        data = data.to(args.device)
        bs = data.y.size(0)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()

        y_true += data.y.data.tolist()
        y_pred += out.max(1)[1].data.tolist()
        y_score += out[:, 1].data.tolist()
        total += bs

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
            loss / total, auc, prec, rec, f1)

    if return_best_thr:
        log_desc = "valid_"
    else:
        log_desc = "test_"
    tensorboard_logger.log_value(log_desc + 'loss', loss / total, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return [prec, rec, f1, auc], loss / len(loader.dataset), best_thr
    else:
        return [prec, rec, f1, auc], loss / len(loader.dataset), None


min_loss = 1e10
patience = 0
best_thr = None

# test
val_metrics, val_loss, thr = test(model, -1, val_loader, return_best_thr=True)
print("Validation loss:{}\teval metrics:".format(val_loss), val_metrics)
test_acc, test_loss, _ = test(model, -1, test_loader, thr=thr)
print("Test performance:", test_acc)
last_epoch = -1

for epoch in range(args.epochs):
    model.train()
    losses_train = []
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        if i % 10 == 0:
            print("Training loss:{}".format(loss.item()))
        loss.backward()
        losses_train.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    tensorboard_logger.log_value('train_loss', np.mean(losses_train), epoch + 1)

    val_metrics, val_loss, thr = test(model, epoch, val_loader, return_best_thr=True)
    print("Validation loss:{}\teval metrics:".format(val_loss), val_metrics)
    test_acc, test_loss, _ = test(model, epoch, test_loader, thr=best_thr)
    print("Test performance:", test_acc)
    if val_loss < min_loss:
        torch.save(model.state_dict(), 'latest.pth')
        print("Model saved at epoch {}".format(epoch))
        min_loss = val_loss
        best_thr = thr
        patience = 0
        logger.info("**************BEST UNTIL NOW*****************")
    else:
        patience += 1
    last_epoch += 1
    if patience > args.patience:
        break

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc, test_loss, _ = test(model, last_epoch+2, test_loader, thr=best_thr)
print("Test performance:", test_acc)
