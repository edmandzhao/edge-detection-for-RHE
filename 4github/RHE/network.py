
import torch.nn.functional as F
import torchgeometry as tgm
from update import GMA
from extractor import BasicEncoderQuarter
from corr import CorrBlock, TransCorrBlock
from utils import *

from einops import rearrange
import numbers
import pidinet
import torchvision.transforms as transforms


autocast = torch.cuda.amp.autocast


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x).contiguous()), h, w).contiguous()




class attHead_image(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """

    def __init__(self, C, C1, layernorm_type='BiasFree', bias=False):
        """
        C: input channel
        C1: output channel after 1*1 conv

        """
        super(attHead_image, self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x1 = self.conv1(x)  # (B, C1, H, W)
        x1 = self.conv2(x1)  # (B, C1, H, W)
        return x1


class EEM1(nn.Module):
    def __init__(self, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image
        C1: input channel of edge
        C2 : output channel of imhead/ehead
        """

        super(EEM1, self).__init__()

        self.imhead = attHead_image(128, 2 * C2)
        self.ehead = nn.Conv2d(240, C2, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)

        self.num_heads = num_heads
        self.project_out = nn.Conv2d(C2, C2, kernel_size=1, bias=bias)
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))

    def edge_att(self, x, e):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """

        _, _, H, W = x.shape

        q1 = self.imhead(x)  # (B, 2*C2, H, W)
        k_eg = self.ehead(e)  # (B, C2, H, W)
        q_im, v_im = q1.chunk(2, dim=1)

        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)

        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1  # (B, head, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        # skip connection
        out = x + self.project_out(out)  # (B, 2, H, W)

        return out.contiguous()

    def forward(self, x, e):
        return self.edge_att(x, e)

class EEM2(nn.Module):
    def __init__(self, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image
        C1: input channel of edge
        C2 : output channel of imhead/ehead
        """

        super(EEM2, self).__init__()

        self.imhead = attHead_image(128, 2 * C2)
        self.ehead = nn.Conv2d(120, C2, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)

        self.num_heads = num_heads
        self.project_out = nn.Conv2d(C2, C2, kernel_size=1, bias=bias)
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))

    def edge_att(self, x, e):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """

        _, _, H, W = x.shape
        q1 = self.imhead(x)  # (B, 2*C2, H, W)
        k_eg = self.ehead(e)  # (B, C2, H, W)
        q_im, v_im = q1.chunk(2, dim=1)

        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)
        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1  # (B, head, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        # skip connection
        out = x + self.project_out(out)  # (B, 2, H, W)

        return out.contiguous()

    def forward(self, x, e):
        return self.edge_att(x, e)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变



        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)



class IHN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args
        self.hidden_dim = 128
        self.context_dim = 128

        self.hidden_dim = 128
        self.context_dim = 128
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            normalize])

        self.fnet1 = BasicEncoderQuarter(output_dim=128, norm_fn='instance')

        self.update_block_4 = GMA(self.args, 32)

        self.update_block_2 = GMA(self.args, 64)

        self.EN1 = pidinet.pidinet()
        self.EN2 = pidinet.pidinet()

        self.EAT1 = EEM1(128, 4, bias=False)
        self.EAT2 = EEM2(128, 4, bias=False)

        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(128)

        self.ehead1 = nn.Conv2d(368, 128, kernel_size=1)
        self.ehead2 = nn.Conv2d(248, 128, kernel_size=1)

    def get_flow_now_4(self, four_point):
        four_point = four_point / 4
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3] - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2] - 1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3] - 1, self.sz[2] - 1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3] - 1, steps=self.sz[3]),
                                      torch.linspace(0, self.sz[2] - 1, steps=self.sz[2]))

        points = torch.cat(
            (gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
            dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def get_flow_now_2(self, four_point):
        four_point = four_point / 2
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3] - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2] - 1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3] - 1, self.sz[2] - 1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3] - 1, steps=self.sz[3]),
                                      torch.linspace(0, self.sz[2] - 1, steps=self.sz[2]))
        points = torch.cat(
            (gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
            dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow

    def get_H(self, four_point):
        four_point = four_point / 2
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3] - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2] - 1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3] - 1, self.sz[2] - 1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        return H

    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 4, W // 4).to(img.device)
        coords1 = coords_grid(N, H // 4, W // 4).to(img.device)

        return coords0, coords1

    def initialize_flow_2(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 2, W // 2).to(img.device)
        coords1 = coords_grid(N, H // 2, W // 2).to(img.device)

        return coords0, coords1

    def forward(self, image1, image2, img1_g, img2_g, iters_lev0=6, iters_lev1=3, test_mode=False):


        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()  # 深拷贝，相当于复制了一份
        image2 = image2.contiguous()

        e11, x11,x12 = self.EN1(img1_g)
        e1 = e11[-1]
        e1 = e1.float()

        e22,x21,x22 = self.EN2(img2_g)
        e2 = e22[-1]




        fmap2o = None
        with autocast(enabled=self.args.mixed_precision):
            fmap1_32, fmap1_64 = self.fnet1(image1)
            fmap2_32, _ = self.fnet1(image2)
        fmap1 = fmap1_32.float()
        fmap2 = fmap2_32.float()


        fmap1 = self.EAT1(fmap1,x12)
        fmap2 = self.EAT1(fmap2,x22)

        fmap1 = self.cbam1(fmap1)
        fmap2 = self.cbam1(fmap2)

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, sz=32)
        coords0, coords1 = self.initialize_flow_4(image1)  # 这里取一样的是初始化，因为第一个H是1，也就是不变化

        sz = fmap1_32.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)
        bz = coords0.shape[0]
        flow1 = torch.zeros(6, bz, 2, 2, 2)
        flow2 = torch.zeros(iters_lev1, bz, 2, 2, 2)

        for itr in range(iters_lev0):
            corr = corr_fn(coords1)  # 把转换过去的放进去，其实就是x'放进去，得到的就是sk
            flow = coords1 - coords0  # 算的单应流

            with autocast(enabled=self.args.mixed_precision):
                if self.args.weight:
                    delta_four_point, weight = self.update_block_4(corr, flow)
                else:
                    delta_four_point = self.update_block_4(corr, flow)  # GMA 输出的是偏移

            four_point_disp = four_point_disp + delta_four_point  # 更新结果
            flow1[itr] = four_point_disp
            coords1 = self.get_flow_now_4(four_point_disp)





        four_point_disp_med = four_point_disp
        flow_med = coords1 - coords0
        flow_med_1 = F.upsample_bilinear(flow_med, None, [2, 2]) * 2
        flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4

        image2 = warp(image2, flow_med)
        x21w = warp(x21, flow_med_1)



        with autocast(enabled=self.args.mixed_precision):
            _, fmap2_64 = self.fnet1(image2)


        fmap1 = fmap1_64.float()
        fmap2 = fmap2_64.float()


        fmap1 = self.EAT2(fmap1, x11)
        fmap2 = self.EAT2(fmap2, x21w)

        fmap1 = self.cbam2(fmap1)
        fmap2 = self.cbam2(fmap2)


        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, sz=32)

        coords0, coords1 = self.initialize_flow_2(image1)
        sz = fmap1.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)



        for itr in range(iters_lev1):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.weight:
                    delta_four_point, weight = self.update_block_2(corr, flow)
                else:
                    delta_four_point = self.update_block_2(corr, flow)
            four_point_disp = four_point_disp + delta_four_point
            flow2[itr] = four_point_disp + four_point_disp_med
            coords1 = self.get_flow_now_2(four_point_disp)



        four_point_disp = four_point_disp + four_point_disp_med

        return four_point_disp, flow1, flow2,e1,e2





