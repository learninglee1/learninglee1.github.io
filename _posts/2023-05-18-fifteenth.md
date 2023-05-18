---
layout: single
title:  "PyTorch 전이학습을 통한 이미지 분류 (15) "
---


# PyTorch

```python
!sudo apt-get install -y fonts-nanum* | tail -n 1
!sudo fc-cache -fv
!rm -rf ~/.cache/matplotlib


# 필요 라이브러리 설치
!pip install torchviz | tail -n 1
!pip install torchinfo | tail -n 1
w = !apt install tree
print(w[-2])


# 라이브러리 임포트
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# 폰트 관련 용도
import matplotlib.font_manager as fm
# 나눔 고딕 폰트의 경로 명시
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = fm.FontProperties(fname=path, size=10).get_name()

# 파이토치 관련 라이브러리
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# warning 표시 끄기
import warnings
warnings.simplefilter('ignore')
# 기본 폰트 설정
plt.rcParams['font.family'] = font_name
# 기본 폰트 사이즈 변경
plt.rcParams['font.size'] = 14
# 기본 그래프 사이즈 변경
plt.rcParams['figure.figsize'] = (6,6)
# 기본 그리드 표시
# 필요에 따라 설정할 때는, plt.grid()
plt.rcParams['axes.grid'] = True
# 마이너스 기호 정상 출력
plt.rcParams['axes.unicode_minus'] = False

# 넘파이 부동소수점 자릿수 표시
np.set_printoptions(suppress=True, precision=4)

# GPU 디바이스 할당
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 공통 함수 다운로드
!git clone https://github.com/wikibook/pythonlibs.git
# 공통 함수 불러오기
from pythonlibs.torch_lib1 import *
# 공통 함수 확인
print(README)

# 데이터 다운로드
w = !wget -nc https://download.pytorch.org/tutorial/hymenoptera_data.zip
# 결과 확인
print(w[-2])

# 압축 해제
w = !unzip -o hymenoptera_data.zip
# 결과 확인
print(w[-1])

# 트리 구조 출력
!tree hymenoptera_data

# Transforms 정의
# 검증 데이터 : 정규화
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# 훈련 데이터 : 정규화에 반전과 RandomErasing 추가
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

# 베이스 디렉터리
data_dir = 'hymenoptera_data'
# 훈련 데이터 디렉터리와 검증 데이터 디렉터리 지정
import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# join 함수 결과 확인
print(train_dir, test_dir)

# 분류하려는 클래스의 리스트 작성
classes = ['ants', 'bees']

# 데이터셋 정의
# 훈련용
train_data = datasets.ImageFolder(train_dir, 
            transform=train_transform)
# 훈련 데이터 이미지 출력용
train_data2 = datasets.ImageFolder(train_dir, 
            transform=test_transform)
# 검증용
test_data = datasets.ImageFolder(test_dir, 
            transform=test_transform)

# 데이터 건수 확인
print(f'훈련 데이터 : {len(train_data)} 건')
print(f'검증 데이터 : {len(test_data)} 건')

# 검증 데이터　
# 처음 10개와 마지막 10개 이미지 출력
plt.figure(figsize=(15, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    image, label = test_data[i]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 11)
    image, label = test_data[-i-1]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1)/2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 데이터로더 정의
batch_size = 10
# 훈련용
train_loader = DataLoader(train_data, 
      batch_size=batch_size, shuffle=True)
# 검증용
test_loader = DataLoader(test_data, 
      batch_size=batch_size, shuffle=False)
# 이미지 출력용
train_loader2 = DataLoader(train_data2, 
      batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data, 
      batch_size=50, shuffle=True)


# 검증 데이터(50건)
torch_seed()
show_images_labels(test_loader2, classes, None, None)

# 파인 튜닝의 경우
# 사전 학습 모델 불러오기
# VGG-19-BN 모델을 학습이 끝난 파라미터와 함께 불러오기
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)
# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)
# 학습률
lr = 0.001
# 손실 함수 정의
criterion = nn.CrossEntropyLoss()
# 최적화 함수 정의
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
# history 파일도 동시에 초기화
history = np.zeros((0, 5))


# 학습
num_epochs = 200
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)

# 결과 확인
evaluate_history(history)

# 난수 고정
torch_seed()

# 검증 데이터 결과 출력
show_images_labels(test_loader2, classes, net, device)


#전이학습의 경우
# VGG-19-BN 모델을 학습이 끝난 파라미터와 함께 불러오기
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 모든 파라미터의 경사 계산을 OFF로 설정
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# 최종 노드의 출력을 2로 변경
# 이 노드에 대해서만 경사 계산을 수행하게 됨
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수로 교차 엔트로피 사용
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))

# 학습
num_epochs = 200
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)

# 결과 확인
evaluate_history(history)

# 난수 고정
torch_seed()

# 검증 데이터 결과 출력
show_images_labels(test_loader2, classes, net, device)
```
