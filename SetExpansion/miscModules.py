# contains misc modules use to construct models
# author: satwik kottur

import torch
import torch.nn as nn
import pdb
from SubLayers import MultiHeadAttention

# identity module
class Identity(nn.Container):
    def __init__(self):
        super(Identity, self).__init__();

    def forward(self, inTensor):
        return inTensor;

# module to split at a given point and sum
class SplitSum(nn.Container):
    def __init__(self, splitSize):
        super(SplitSum, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize];
            secondHalf = inTensor[:, self.splitSize:];
        else:
            firstHalf = inTensor[:, :, :self.splitSize];
            secondHalf = inTensor[:, :, self.splitSize:];
        return firstHalf + secondHalf;

# module to split at a given point and sum
class SplitInfer(nn.Container):
    def __init__(self, splitSize):
        super(SplitInfer, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize];
            secondHalf = inTensor[:, self.splitSize:];
            subtraction = firstHalf - secondHalf
            multiplication = firstHalf * secondHalf
            result = torch.cat((firstHalf, secondHalf, subtraction, multiplication), 1)
        else:
            firstHalf = inTensor[:, :, :self.splitSize];
            secondHalf = inTensor[:, :, self.splitSize:];
            subtraction = firstHalf - secondHalf
            multiplication = firstHalf * secondHalf
            result = torch.cat((firstHalf, secondHalf, subtraction, multiplication), 2)
        return result

class LocalPoolAtt(nn.Module):
    def __init__(self, hiddenSize):
        super(LocalPoolAtt, self).__init__()
        self.selfatt = MultiHeadAttention(1, hiddenSize, hiddenSize, hiddenSize)
    def forward(self, inTensor):
        #if inTensor.dim()==2:
        #    pool = torch.max(inTensor, 0)[0]
        #    pool = pool.expand(inTensor.shape[0],-1)
        #    att, _  = self.selfatt(inTensor, inTensor, inTensor)
        #    result = torch.cat((inTensor, pool, att), 1)
        #else:
        #print(inTensor.shape)
        #print(inTensor.shape)
        #pool = torch.max(inTensor, 1)[0]
        pool = torch.mean(inTensor, 1)
        #print(pool.shape)
        pool = pool.expand(inTensor.shape[1], inTensor.shape[0],-1)
        #print(pool.shape)
        pool = pool.transpose(0,1)
        #print(inTensor.shape)
        att, _  = self.selfatt(inTensor, inTensor, inTensor)
        #print(att.shape)
        result = torch.cat((inTensor, inTensor), 2)
        #print(result.shape)
        return att




# module to split at a given point and sum
class Multimodal(nn.Container):
    def __init__(self, splitSize):
        super(Multimodal, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            instEmbed = inTensor[:, :self.splitSize[0]];
            setEmbed = inTensor[:, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, self.splitSize[1]:];
            concatDim = 1;
        else:
            instEmbed = inTensor[:, :, :self.splitSize[0]];
            setEmbed = inTensor[:, :, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, :, self.splitSize[1]:];
            concatDim = 2;

        return torch.cat((setEmbed + instEmbed, imgEmbed), concatDim);

# module to split at a given point and sum
class MultimodalSum(nn.Container):
    def __init__(self, splitSize):
        super(MultimodalSum, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            instEmbed = inTensor[:, :self.splitSize[0]];
            setEmbed = inTensor[:, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, self.splitSize[1]:];
        else:
            instEmbed = inTensor[:, :, :self.splitSize[0]];
            setEmbed = inTensor[:, :, self.splitSize[0]:self.splitSize[1]];
            imgEmbed = inTensor[:, :, self.splitSize[1]:];

        return setEmbed + instEmbed + imgEmbed;

# module to split at a given point and max
class SplitMax(nn.Container):
    def __init__(self, splitSize):
        super(SplitMax, self).__init__();
        self.splitSize = splitSize; # where to split

    def forward(self, inTensor):
        # Split along particular dimension
        # If only two dims
        if inTensor.dim() == 2:
            firstHalf = inTensor[:, :self.splitSize];
            secondHalf = inTensor[:, self.splitSize:];
        else:
            firstHalf = inTensor[:, :, :self.splitSize];
            secondHalf = inTensor[:, :, self.splitSize:];
        numDims = firstHalf.dim();
        concat = torch.cat((firstHalf.unsqueeze(numDims), \
                            secondHalf.unsqueeze(numDims)), numDims);
        maxPool = torch.max(concat, numDims)[0];
        # satwik: edits for older pytorch version
        #maxPool = torch.max(concat, numDims)[0].squeeze(numDims);
        return maxPool;

# Module to nullify the inputs, ie fill them with zeros
class Nullifier(nn.Container):
    def __init__(self):
        super(Nullifier, self).__init__();

    def forward(self, inTensor):
        outTensor = inTensor.clone();
        outTensor.fill_(0.0);
        return outTensor;
