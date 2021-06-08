import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from config import BATCH_SIZE, PROPOSAL_NUM, test_model
from core import model, dataset
from torch.nn import DataParallel
from torch.autograd import Variable
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from torchvision.utils import make_grid
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs

def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_feature(img,model,dst = './feautures'):
    # pic_dir = './images/2.jpg'
    # transform = transforms.ToTensor()
    # img = get_picture(pic_dir, transform)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # 插入维度
    # img = img.unsqueeze(0)
    #
    # img = img.to(device)

    # net = models.resnet101().to(device)
    # net.load_state_dict(torch.load('./model/resnet101-5d3b4d8f.pt'))
    exact_list = None
    make_dirs(dst)
    therd_size = 256

    myexactor = FeatureExtractor(model, exact_list)
    outs = myexactor(img)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        if 'fc' in k:
            continue
        features_t=features.unsqueeze(0).permute((1, 0, 2, 3))
        im_all=make_grid(features_t, nrow=int(math.sqrt(features_t.size(0))),padding=0).permute((1, 2, 0))
        im_all=(im_all.data.numpy()*255.).astype(np.uint8)
        im_all = cv2.applyColorMap(im_all, cv2.COLORMAP_JET)
        Image.fromarray(im_all).save(dst+'/'+str(k) + '.jpg')


        for i in range(1):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            feature = features.data.numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            dst_file = os.path.join(dst_path, k + '_' + str(i) + '.png')
            if feature_img.shape[0] < therd_size:
            #    tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(dst_file, tmp_img)
            else:
                cv2.imwrite(dst_file, feature_img)
    plt.show()


os.environ['KMP_DUPLICATE_LIB_OK']='True'
image_path='/Users/zhulingang/Desktop/新菊花数据集/test/070.钟山明樱/IMG_3251.jpeg'
to_tensor=transform=transforms.Compose([
                                                transforms.Resize(448),
                                                transforms.CenterCrop(448),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                                ])
img=Image.open(image_path)
fig2=plt.figure()
#plt.imshow(img)
img=to_tensor(img)
img = torch.unsqueeze(img, 0)
net = model.attention_net(topN=PROPOSAL_NUM)
ckpt = torch.load(test_model, map_location='cpu')
net.load_state_dict(ckpt['net_state_dict'])
#net = net.cuda()
#net = DataParallel(net)
criterion = nn.CrossEntropyLoss()
net.eval()
get_feature(img, net.pretrained_model)
_, concat_logits, _, _, _,parts_images,concat_out = net(img)
#TODO 画分类柱状图
print(concat_logits.shape)
x=[ i for i in range(115)]
concat_logits=F.softmax(concat_logits,dim=1)
plt.bar( x=x,height=concat_logits.data.squeeze(0).numpy(),linewidth=1)
plt.show()

_, concat_predict = torch.max(concat_logits, 1)

#TODO 画局部区域特征向量
# concat_f=concat_out.view(1,128,-1)
# concat_f=concat_f.squeeze(0)
# concat_f=(concat_f.data.numpy()*255.).astype(np.uint8)
# plt.xticks([])  # 去掉横坐标值
# plt.yticks([])  # 去掉纵坐标值
# plt.imshow(concat_f)
# plt.show()
# plt.imsave('./concat_feature.jpg',concat_f)
# #Image.fromarray(concat_f).save('concat_feature'+ '.jpg')

#TODO 画局部区域卷积图
for i in range(model.CAT_NUM):
    get_feature(parts_images[i].unsqueeze(0), net.pretrained_model, './feautures'+str(i))


#Todo 将局部区域图显示
parts_all=make_grid(parts_images,nrow=3).permute((1, 2, 0))
parts_all=(parts_all.detach().data.numpy()*255.).astype(np.uint8)
#parts_all = cv2.applyColorMap(parts_all, cv2.COLORMAP_JET)
Image.fromarray(parts_all).save('part'+ '.jpg')
fig=plt.figure()
for i in range(parts_images.size(0)):
    img = parts_images[i]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    img = img.numpy()  # FloatTensor转为ndarray
    img[img < 0]=0
    img[img > 1]=1
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    temp=fig.add_subplot(2,3,i+1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    temp=plt.imshow(img)
    plt.imsave('./parts/'+str(i)+'.jpg',img)
plt.savefig('testblueline.jpg')
plt.show()

# outputs = F.softmax(outputs, dim=1)
# predicted = torch.max(outputs, dim=1)[1].cpu().item()
print(concat_predict.data.item())
