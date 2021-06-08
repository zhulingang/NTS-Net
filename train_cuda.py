import os
import torch.utils.data

from torch.nn import DataParallel
import torchvision.transforms as transforms
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model_cuda, dataset
from core.utils import init_log
from core import data
from tensorboardX import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#如果有两块可以设置0 、1
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info
writer1=SummaryWriter('./runs/train_loss')
writer2=SummaryWriter('./runs/test_acc')
# read dataset
# trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                           shuffle=True, num_workers=8, drop_last=False)
# testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                          shuffle=False, num_workers=8, drop_last=False)
#
trainset = data.MyDataset('./CUB200/new_train.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=0)

testset = data.MyDataset('./CUB200/test.txt', transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)
# define model
net = model_cuda.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[40, 80], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[40, 80], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[40, 80], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[40, 80], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, 90):
    # begin training
    _print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        niter = epoch * len(trainloader) + i
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()
        # raw_logits为原图的分类特征向量，concat为原图和局部区域生成的特征向量，part_logits为局部区域的分类特征向量
        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        #part_loss计算置信度
        part_loss = model_cuda.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                         label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        #raw_loss是原图的特征分类向量与正确类别的交叉熵
        raw_loss = creterion(raw_logits, label)
        #是合并特征经全连接层输出的特征分类向量与正确类别的交叉熵
        concat_loss = creterion(concat_logits, label)#Scrutinizer损失值
        #top_n_prob为信息量，part_loss为置信度
        rank_loss = model_cuda.ranking_loss(top_n_prob, part_loss)#navigator网络损失，比较信息度和置信度排序，计算损失值
        #为为局部区域的分类特征向量与正确类别的交叉熵
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))#局部区域的损失值，即Teacher网络=局部区域+整个图像

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        writer1.add_scalars('Train_loss', {'train_loss': total_loss.data.item()}, niter)
#        progress_bar(i, len(trainloader), 'train')
    #调整学习率   
    for scheduler in schedulers:
        scheduler.step()
    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
#                progress_bar(i, len(trainloader), 'eval train set')

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

	# evaluate on test set
        niter_test = epoch * len(trainloader)
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
#                progress_bar(i, len(testloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        writer2.add_scalars('Test_acc', {'test_acc': 100.0 * float(test_acc)}, niter_test)
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')
