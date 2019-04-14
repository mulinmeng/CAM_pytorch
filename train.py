import torch
import numpy as np
from torchvision import models, transforms
import torch.optim as optim
import models
import torch.backends.cudnn as cudnn
import argparse
import os
import random
from torch.autograd import Variable

from dataset import PASCALVOC
import utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--epoch', type=int, default=100, help='training epoches')
parser.add_argument('--data_dir', type=str, required=True, help='parameters storage')
parser.add_argument('--cuda', action='store_true', help='use GPU to train')
parser.add_argument('--img_size', type=int, default=321, help='image size')
parser.add_argument('--num_class', type=int, default=20, help='label classes')
parser.add_argument('--log_interval', type=int, default=20, help='log messages interval')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# fix the random seed
random_seed = np.random.randint(0, 1000000)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
param_dir = "param_dir"
# Data
print('==> Preparing data..')

trainset = PASCALVOC(
    data_dir=args.data_dir,
    imageset="train",
    devkit="./devkit"
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2
)

testset = PASCALVOC(
    data_dir=args.data_dir,
    imageset='val',
    devkit="./devkit"
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2)

log_dir = 'output/log/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

net = models.vgg16(pretrained=True)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])

criterion = utils.MultiSigmoidCrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
net.train()
logger = utils.Logger(stdio=True, log_file=log_dir+"training.log")
images = Variable(torch.FloatTensor(args.batch_size, 3, args.img_size, args.img_size))
labels = Variable(torch.FloatTensor(args.batch_size, args.num_class))
# use cuda
if args.cuda:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    criterion = criterion.cuda()
    images = images.cuda()
    labels = labels.cuda()


def load_data(v, data):
    v.data.resize_(data.size()).copy_(data)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    for batch_idx, (img, lbl, shapes) in enumerate(trainloader):
        load_data(images, img)
        load_data(labels, lbl)
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        print(train_loss)


logger.log('starting to train')
iter_steps = 0
for epoch in range(args.epoch):
    net.train()
    train(epoch)
    if (epoch + 1) % args.val_interval == 0:
        pass
    # if (epoch+1) % opt.save_interval == 0:
    torch.save(net.state_dict(), os.path.join(param_dir, 'epoch_{}.pth'.format(epoch)))
    torch.save(net.state_dict(), os.path.join(param_dir, 'resume.pth'))









