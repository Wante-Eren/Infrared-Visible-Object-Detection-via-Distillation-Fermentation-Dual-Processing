import torch
import torch.nn as nn
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


    
class DEA(nn.Module):
    """Dynamic Enhancement Attention (DEA) module.
       x0 --> RGB feature map,  x1 --> IR feature map
    """
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        assert channel % reduction == 0, "Input channel size must be divisible by reduction"
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x[0] - RGB Feature map | x[1] - IR Feature map
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)

class DECA(nn.Module):
    """Dynamic Enhancement Channel Attention (DECA) module.
       x0 --> RGB feature map,  x1 --> IR feature map
    """
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.channel = channel
        self.kernel_size = kernel_size

        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully-connected layer for attention
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Activation and compression network
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)  # Compress concatenated features

        # Multi-scale convolution pyramid kernels
        if p_kernel is None:
            p_kernel = [5, 4]  # Default pyramid kernels
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=1), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=1), nn.SiLU())
        self.conv_c3 = None  # Dynamically assigned during forward

    def forward(self, x):
        """Forward pass for DECA.
        Args:
            x[0]: RGB Feature map, shape=(B, C, H, W)
            x[1]: IR Feature map, shape=(B, C, H, W)
        Returns:
            torch.Tensor: Concatenated result_vi and result_ir
        """
        b, c, h, w = x[0].size()

        # Convolution pyramid mechanism
        glob_t = self.compress(torch.cat([x[0], x[1]], dim=1))  # Concatenate RGB & IR (channels doubled)

        # Dynamically calculate convolution kernel size
        kernel_size = int(min(h, w) * 0.5)
        dynamic_kernel = max(1, min(kernel_size, min(h, w)))  # Prevent kernel > input size

        # Check and create convolution kernel dynamically
        if h >= dynamic_kernel and w >= dynamic_kernel:
            self.conv_c3 = nn.Sequential(
                nn.Conv2d(self.channel, self.channel, dynamic_kernel, dynamic_kernel, 0, groups=1),
                nn.SiLU()
            )
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))  # Apply convolution
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)  # Fallback pooling if input is too small

        # Compute attention weights
        w_vi = self.avg_pool(x[0]).view(b, c)
        w_ir = self.avg_pool(x[1]).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        # Generate result_vi and result_ir
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return torch.cat([result_vi, result_ir], dim=1)

class DEPA(nn.Module):
    """Dynamic Enhancement Pixel Attention (DEPA)
       x0 --> RGB feature map,  x1 --> IR feature map
    """
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.channel = channel
        if m_kernel is None:
            m_kernel = [3, 7]

        # Individual convolution blocks for RGB and IR
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)

        # Merging convolutions
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

        self.act = nn.Sigmoid()

    def forward(self, x):
        """Forward pass for DEPA."""
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], dim=1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], dim=1))
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))

        # Compute pixel-level attention
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)

        # Apply pixel-level attention
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir