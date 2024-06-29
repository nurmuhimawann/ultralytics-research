import torch
import torch.nn as nn
try:
    from pytorch_wavelets import DWTForward # 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
                                            # 如果提示缺少pywt库则安装 pip install PyWavelets
except:
    pass

__all__ = ['Down_wt']

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = Down_wt(64, 32)

    out = model(image)
    print(out.size())