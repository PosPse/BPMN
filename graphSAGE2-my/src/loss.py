import torch.nn as nn
import torch

class Loss(nn.Module):
    def __init__(self, rate):
        super(Loss, self).__init__()
        self.rate = rate
    def forward(self, x, label):
        class_output = []
        #权重动态计算更好
        cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.rate))
        label = torch.tensor(label)
        '''
        # 如果是测试，输出分类结果
        if self.mode == 'test':
            x = x.squeeze(0)
            padding_index = -1
            for j in label[0]:
                if j.equal(torch.LongTensor([0, 0, 0])):
                    break
                else:
                    padding_index += 1
            for i in range(0,padding_index+1):
                max_index = 0
                if x[i][1] >= x[i][max_index]:
                    max_index = 1
                if x[i][2] >= x[i][max_index]:
                    max_index = 2
                if max_index == 0:
                    class_output.append(0)
                if max_index == 1:
                    class_output.append(1)
                if max_index == 2:
                    class_output.append(2)
            x = x.unsqueeze(0)
            assess(x[0][:padding_index+1],label[0][:padding_index+1])
        '''
        temp_loss = cross_entropy(x, label)
        loss = temp_loss
        return loss, class_output

