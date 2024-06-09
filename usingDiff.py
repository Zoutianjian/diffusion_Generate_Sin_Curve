import torch
import numpy as np
import torch.nn as nn
import copy
import random
length = 512
length_train = 512
periodMin = 60 # 原版是72
periodMax = 60
t = 2000
betaList = np.clip((np.arange(t)-(t/2))/(t/2)*0.15,a_min=0.005,a_max=0.01)
a = 1 - betaList
a_para = np.array([np.exp(np.sum(np.log(a[0:ii+1]))) for ii in range(t)])
timeEmbedding = 32
periodEmbedding = 16

class SingleSignalBottomLayer(nn.Module):
    def __init__(self,inputDim=64,outputDim=64):
        super(SingleSignalBottomLayer,self).__init__()
        self.activate_func = nn.ReLU()
        
        self.ConvS = nn.Sequential(nn.Conv1d(in_channels=inputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
                    self.activate_func)
        self.ConvRes = nn.Sequential(
            nn.Conv1d(in_channels=outputDim+timeEmbedding,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func,
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
        )
        self.ConvE = nn.Sequential(
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func)
        # self.attentionBlock = nn.TransformerEncoderLayer(d_model=outputDim,nhead=outputDim,dim_feedforward=outputDim,batch_first=True)
        # Transformer 会更加浪费显存
        self.attentionBlock = nn.Sequential(
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func)

    def forward(self,timeEmbed:torch.Tensor,input:torch.Tensor):
        input = self.ConvS(input)
        forward_1 = self.activate_func(input+self.ConvRes(torch.cat([timeEmbed,input],dim=1)))
        # forward_2 = self.ConvE(forward_1 + self.attentionBlock(forward_1.transpose(1,2)).transpose(1,2))
        forward_2 = self.ConvE(forward_1 + self.attentionBlock(forward_1))
        return forward_2
    
class SingleSignalDown(nn.Module):
    def __init__(self,inputDim=16,outputDim=32):
        super(SingleSignalDown,self).__init__()
        self.activate_func = nn.ReLU()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.Conv1S = nn.Sequential(nn.Conv1d(in_channels=inputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func
        )
        
        self.Conv1E = nn.Sequential(
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func)
        
        self.Conv2S = nn.Sequential(
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func)
        
        self.Conv2E = nn.Sequential(
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func)
        self.ConvRes1 = nn.Sequential(
            nn.Conv1d(in_channels=outputDim+timeEmbedding,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func,
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
        )
        self.ConvRes2 = nn.Sequential(
            nn.Conv1d(in_channels=outputDim+timeEmbedding,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
            self.activate_func,
            nn.Conv1d(in_channels=outputDim,out_channels=outputDim,kernel_size=3,stride=1,padding='same'),
        )
    def forward(self,timeEmbed:torch.Tensor,input:torch.Tensor):
        forward_0 = self.Conv1S(input)
        forward_1 = self.activate_func(forward_0+self.ConvRes1(torch.cat([forward_0,timeEmbed],dim=1)))
        forward_1 = self.Conv1E(forward_1)
        forward_2 = self.Conv2S(forward_1)
        forward_2 = self.activate_func(
            forward_2+self.ConvRes2(torch.cat([forward_2,timeEmbed],dim=1)))
        forward_2 = self.Conv2E(forward_2)
        return input,forward_1,forward_2

class SingleSignalUp(nn.Module):
    def __init__(self,upSampleDim=32,inputDimList=[16,16,16],outputDimList=[16,16,16]):
        super(SingleSignalUp,self).__init__()
        self.upSampleDim = upSampleDim
        self.inputDimList = inputDimList
        self.outputDimList = outputDimList
        self.Decoder1 = SingleSignalBottomLayer(upSampleDim+inputDimList[2],outputDimList[2])
        self.Decoder2 = SingleSignalBottomLayer(inputDimList[1]+outputDimList[2],outputDimList[1])
        self.Decoder3 = SingleSignalBottomLayer(inputDimList[0]+outputDimList[1],outputDimList[0])
    
    def forward(self,timeEmbed:torch.Tensor,initialTensor:torch.Tensor,inputTensor1:torch.Tensor,
            inputTensor2:torch.Tensor,inputTensor3:torch.Tensor):
        forward0 = torch.cat([initialTensor,inputTensor1],dim=1)
        forward1 = self.Decoder1(timeEmbed,forward0)
        forward1 = torch.cat([forward1,inputTensor2],dim=1)
        forward2 = self.Decoder2(timeEmbed,forward1)
        forward2 = torch.cat([forward2,inputTensor3],dim=1)
        forward3 = self.Decoder3(timeEmbed,forward2)
        return forward3
    
class DiffusionSin(nn.Module):
    def __init__(self,LayerNum=4,initialLayers=16,finalLayers=16,
        outputDimList = [16,32,48,64]):
        super(DiffusionSin,self).__init__()
        self.activatefunc = nn.SiLU()
        self.timeEmbedding = nn.Embedding(t+1,timeEmbedding-periodEmbedding)
        self.periodEmbedding = nn.Embedding(int(length_train/2)+1,periodEmbedding)
        self.initialLayer =nn.Conv1d(in_channels=7,out_channels=initialLayers,kernel_size=3,stride=1,padding='same')
        self.finalLayer = nn.Conv1d(in_channels=finalLayers+6,out_channels=1,kernel_size=3,stride=1,padding='same')
        self.upSample = nn.Upsample(scale_factor=2,mode = 'linear',align_corners=True)
        self.downSampe = nn.MaxPool1d(kernel_size=2,stride=2)
        if LayerNum != len(outputDimList):
            if (len(outputDimList)) > LayerNum:
                outputDimList = outputDimList[0:LayerNum] 
            else:
                outputDimList.extend([outputDimList[-1] for num in range(LayerNum-len(outputDimList))])
        inputList = copy.deepcopy(outputDimList)
        inputList.insert(0,initialLayers)
        outputList = copy.deepcopy(outputDimList)
        outputList.insert(0,finalLayers)
        outputList.append(outputDimList[-1])
        self.EncoderList = nn.ModuleList(
            [ SingleSignalDown(inputList[index],inputList[index+1]) for index in range(LayerNum)]
        )
        self.DecoderList = nn.ModuleList(
            [ SingleSignalUp(outputList[index+2],[inputList[index],inputList[index+1],inputList[index+1]],
                [outputList[index+1],outputList[index+1],outputList[index+1] if index !=0 else outputList[0]]) for index in range(LayerNum)]
        ) # 注意，decoder 是栈结构，需要从后往前依次调用
        self.bottom1 = SingleSignalBottomLayer(outputList[-1],outputList[-1])
        self.bottom2 = SingleSignalBottomLayer(outputList[-1],outputList[-1])
        self.LayerNum = LayerNum
        indexTensor = torch.arange(length,dtype=torch.float32).unsqueeze(0)
        w1,w2,w3 = 2*np.pi/length, 2*np.pi/(length/2), 2*np.pi/(length/4)
        sin1,cos1 = torch.sin(indexTensor*w1),torch.cos(indexTensor*w1)
        sin2,cos2 = torch.sin(indexTensor*w2),torch.cos(indexTensor*w2)
        sin3,cos3 = torch.sin(indexTensor*w3),torch.cos(indexTensor*w3)
        self.posEmbed = torch.cat([sin1,cos1,sin2,cos2,sin3,cos3],dim=0).unsqueeze(0)
        
    def forward(self,timeT:torch.Tensor,noise:torch.Tensor,period:torch.Tensor):
        timeVector = self.timeEmbedding(timeT)
        periodVector = self.periodEmbedding(period)
        posVector = self.posEmbed.clone().repeat((noise.shape[0],1,1)).to(noise.device)
        tensorStack = []
        noise = torch.cat([posVector,noise],dim=1)
        x = self.initialLayer(noise)
        for i in range(self.LayerNum):
            timeVectorNow = timeVector.unsqueeze(0).unsqueeze(2).repeat(x.shape[0],1,x.shape[2])
            periodVectorNow = periodVector.transpose(1,2).repeat(1,1,x.shape[2])
            timeVectorNow = torch.cat([timeVectorNow,periodVectorNow],dim=1)
            forward_0,forward_1,forward_2 = self.EncoderList[i](timeVectorNow,x)
            x = self.downSampe(forward_2)
            tensorStack.append((forward_0,forward_1,forward_2))
        timeVectorNow = timeVector.unsqueeze(0).unsqueeze(2).repeat(x.shape[0],1,x.shape[2])
        periodVectorNow = periodVector.transpose(1,2).repeat(1,1,x.shape[2])
        timeVectorNow = torch.cat([timeVectorNow,periodVectorNow],dim=1)
        x = self.bottom1(timeVectorNow,x)
        x = self.bottom2(timeVectorNow,x)
        for i in range(self.LayerNum-1,-1,-1):
            x = self.upSample(x)
            timeVectorNow = timeVector.unsqueeze(0).unsqueeze(2).repeat(x.shape[0],1,x.shape[2])
            periodVectorNow = periodVector.transpose(1,2).repeat(1,1,x.shape[2])
            timeVectorNow = torch.cat([timeVectorNow,periodVectorNow],dim=1)
            backward_2,backward_1,backward_0 = tensorStack.pop(-1)
            x = self.DecoderList[i](timeVectorNow,x,backward_0,backward_1,backward_2)
        x = torch.cat([posVector,x],dim=1)
        x = self.finalLayer(x)
        return x

device = 'cuda:2'
mainModel = DiffusionSin(
    LayerNum=4,initialLayers=32,finalLayers=32,outputDimList=[32,64,96,128])
modelPath = './Model_Tests/diffEmbedPeriod_Test.pt'
try:
    mainModel.load_state_dict(torch.load(modelPath)) # 单卡训练出来的模型
except:
    mainModel.load_state_dict({k.replace('module.', ''):v for k, v in
                torch.load(modelPath).items()})
mainModel = mainModel.to(device)

dataNums = 1
blankLength = 40
blankNums = 4
starts = [[random.randint(0,length-blankLength-2) for j in range(blankNums)] for i in range(dataNums)]

targetTensor = torch.randn((dataNums,1,length),dtype=torch.float32).to(device)
estimateNoiseLast = targetTensor.clone()
fillBlankTensor = torch.sin( torch.arange(length)*(2*(np.pi)/periodMin) +random.uniform(0,2*np.pi)
                            ).unsqueeze(0).unsqueeze(0).repeat(dataNums,1,1)
period = torch.tensor([random.randint(periodMin/4,periodMax/4)*4 for i in range(dataNums)],dtype=torch.int64).reshape(dataNums,1).to(device)

for idData,blankList in enumerate(starts):
    for blankNumber in blankList:
        fillBlankTensor[idData:idData+1,0:1,blankNumber:blankNumber+blankLength] = torch.randn((1,1,blankLength))
fillBlankTensor = fillBlankTensor.to(device)

for timeT in range(t-1,-1,-1):
    timeTensor = torch.tensor(timeT+1,dtype=torch.long).to(device)
    with torch.no_grad():
        # 这是最简单的生成模式
        estimateNoise = mainModel(timeTensor,targetTensor,period)
        targetTensor = 1/np.sqrt(a[timeT])*(targetTensor-(1-a[timeT])/np.sqrt(1-a_para[timeT])*estimateNoise
            ) + min(betaList[timeT],0.005)*(torch.randn((dataNums,1,length)).to(device))
        # targetTensor = torch.clip(targetTensor,-2,3) # 实际上无需 clip
        
        # 下面测试一下填空模式
        # noisetime = fillBlankTensor*float(np.sqrt(a_para[timeT]))+float(np.sqrt(1-a_para[timeT])
        #             )*torch.randn((dataNums,1,length)).to(device)
        # for idData,blankList in enumerate(starts):
        #     for blankNumber in blankList:
        #         noisetime[idData:idData+1,0:1,blankNumber:blankNumber+blankLength] = targetTensor[
        #             idData:idData+1,0:1,blankNumber:blankNumber+blankLength]
        # targetTensor = noisetime

        # estimateNoiseLast = mainModel(timeTensor,targetTensor,period)
        # targetTensor =  1/np.sqrt(a[timeT])*(targetTensor-(1-a[timeT])/np.sqrt(1-a_para[timeT])*estimateNoiseLast) 
        
from matplotlib import pyplot as plt
targetData = targetTensor.squeeze(1).cpu().numpy()
fillBlankData = fillBlankTensor.squeeze(1).cpu().numpy()
noiseCompare = torch.randn([1,1,512]).squeeze(0).squeeze(0).cpu().numpy()
sinReal = torch.sin(torch.arange(512)*(2*(np.pi))/periodMin).cpu().numpy()
plt.cla()
# plt.plot(noiseCompare,color='orange')
# plt.plot(sinReal,color='green')
# for i in range(dataNums):
#     plt.plot(np.reshape(fillBlankData[i],-1),label='filledTensor%d'%(i))
    
for i in range(dataNums):
    plt.plot(np.reshape(targetData[i],-1),label='targetData%d'%(i))
    
plt.legend()
plt.savefig("./Ball_Points/Sin.png",bbox_inches='tight',dpi = 600)
a = 1
