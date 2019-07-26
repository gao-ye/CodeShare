import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import sys
# sys.path.append(sys.path[0]+'/models/transformer')

import pdb

def adjust_lr_exp(optimizer, base_lr, iter, total_iter):
  
  if iter < total_iter/2:
     local_lr = base_lr
  elif iter < total_iter*3/4:
     local_lr = base_lr * (0.1**1)
  elif iter < total_iter*7/8:
     local_lr = base_lr * (0.1**2)
  elif iter < total_iter*15/16:
     local_lr = base_lr * (0.1**3)
  else:
     local_lr = base_lr * (0.1**4)
     
  if optimizer.param_groups[0]['lr']!=local_lr:
       print('=============> lr adjusted to ',local_lr)
  for g in optimizer.param_groups:
    g['lr'] = local_lr
    
    
class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self._scanned_list = False
        self._out_of_list = ''
        self._ignore_case = True
        self.sep = sep
        self.alphabet = alphabet.split(sep)

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i

    def scan(self, text):
        # print(text)
        text_tmp = text
        text = []
        for i in range(len(text_tmp)):
            text_result = ''
            for j in range(len(text_tmp[i])):
                chara = text_tmp[i][j].lower() if self._ignore_case else text_tmp[i][j]
                if chara not in self.alphabet:
                    if chara in self._out_of_list:
                        continue
                    else:
                        self._out_of_list += chara
                        file_out_of_list = open("out_of_list.txt", "a+")
                        file_out_of_list.write(chara + "\n")
                        file_out_of_list.close()
                        print('" %s " is not in alphabet...' % chara)
                        continue
                else:
                    text_result += chara
            text.append(text_result)
        text_result = tuple(text)
        self._scanned_list = True
        return text_result

    def encode(self, text, scanned=True):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        self._scanned_list = scanned
                
        if not self._scanned_list:
            text = self.scan(text)

        if isinstance(text, str):
            text = [
                #self.dict[char.lower() if self._ignore_case else char]
                EOS if char==' ' else self.dict[char]
                for char in text
            ]
            length = [len(text)]

        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([' ' if i>=BOS else self.alphabet[i] for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l ], torch.LongTensor([l])))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def loadData(v, data):
    # tensor.opt_() 会在原地执行opt操作，并返回改变后的tensor
    v.data.resize_(data.size()).copy_(data)


def label_convert(ori_lab, nclass):
    [batch_size] = ori_lab.shape
    # label = torch.IntTensor(batch_size)
    label = ori_lab.reshape(batch_size,1).type(torch.LongTensor)
    # print(label)
    one_hot = torch.zeros(batch_size, nclass).scatter_(1, label, 1)

    return one_hot
    # print(one_hot) 
