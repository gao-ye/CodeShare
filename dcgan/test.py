import torch.nn as nn
import torch
import cv2
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1=nn.Sequential(
            nn.ConvTranspose2d(#stride(input_w-1)+k-2*Padding
                in_channels=100,
                out_channels=64*8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(64*8),
            nn.ReLU(inplace=True),
 
        )#14
        self.deconv2=nn.Sequential(
            nn.ConvTranspose2d(#stride(input_w-1)+k-2*Padding
                in_channels=64*8,
                out_channels=64*4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64*4),
            nn.ReLU(inplace=True),
 
        )#24
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=64*4,
                out_channels=64*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64*2),
            nn.ReLU(inplace=True),
 
        )#48
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=64*2,
                out_channels=64*1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
 
        )
        self.deconv5=nn.Sequential(
            nn.ConvTranspose2d(64,3,5,3,1,bias=False),
            nn.Tanh(),
        )
    def forward(self, x):
        x=self.deconv1(x)
        x=self.deconv2(x)
        x=self.deconv3(x)
        x=self.deconv4(x)
        x=self.deconv5(x)
        return  x
 
g=torch.load("pkl/15g.pkl")
imgs=g(torch.randn(100,100,1,1).cuda())
for i in range(len(imgs)):
    img=imgs[i].permute(1,2,0).cpu().detach().numpy()*255
    cv2.imwrite("bitmaps/"+str(i)+".jpg",img,)#这里需要在同目录下建立一个bitmap文件夹
print("done")
