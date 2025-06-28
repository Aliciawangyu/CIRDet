
import torch 
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

        
class EFD(nn.Module):
    '''
    the module of UPD 
    '''
    def __init__(self, c1, ratio=16):
        super().__init__()
        self.weight1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )
        self.weight2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, diff, x):
        y = self.weight2(diff) * (self.weight1(diff) * x)
        return y, x - y

class Avg(nn.Module):
    def __init__(self):
        super(Avg, self).__init__()
    
    def forward(self, x1, x2):
        return x1 + x2


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class attention(nn.Module):
    def __init__(self, dim, k=7, num_head=8):
        super().__init__()
        self.k = k
        self.num_head = num_head
        layer_scale_init_value = 1e-6  
        self.proj = nn.Linear(dim//2, dim)
    
        self.short_cut_linear = nn.Linear(dim, dim//2)
        self.l = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.k,self.k))
            
        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.mlp = MLP(dim, mlp_ratio=4)

    def forward(self, x, x_e): #x 为主模态
        res_x = x
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        
        short_cut = x.permute(0, 3, 1, 2)############# B,H,W,C -> B,C,H,W

        x = self.l(x).permute(0, 3, 1, 2) # BCHW
        x = self.act(x)

        b = x.permute(0, 2, 3, 1) # BHWC
        kv = self.kv(b)
        kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4) #2，B，nh, HW, C//nh//2
        k, v = kv.unbind(0) #Wk Wv # [B, nh, HW, C//nh//2]
        short_cut = self.pool(short_cut).permute(0,2,3,1) #[B 7 7 C]
        short_cut = self.short_cut_linear(short_cut) #W_Q  [B 7 7 C//2]
        q = short_cut # [B 7 7 C//2]
        short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
        m = short_cut # [B hw nh C//nh//2]->[B nh hw C//nh//2]
        attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1)  # [B nh hw HW]
        attn = attn.softmax(dim=-1)
        # [B nh hw C//nh//2] --> [B nh k k C//nh//2] -->[B nh C//nh//2 k k] --> [B, C//2 k k]
        attn = (attn @ v).reshape(B, self.num_head, self.k, self.k, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.k, self.k)
        attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # [B H W C//2]
        
        x = self.proj(attn) # B H W C
        x = self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x + res_x
        x = x + self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x)

        return x, q
    

class GFCA(nn.Module):
    def __init__(self, dim, head=4):
        super().__init__()
        self.ir_attn = attention(dim, head)
        self.rgb_attn = attention(dim, head)
        self.conv = Conv(dim, dim, 1, 1)
        self.fc = nn.Linear(dim, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, rgb, ir):
        rgb = rgb.permute(0, 2, 3, 1).contiguous()
        ir = ir.permute(0, 2, 3, 1).contiguous()

        rgb_out, rgb_q = self.rgb_attn(rgb, ir)
        ir_out, ir_q = self.ir_attn(ir, rgb)
       
        B, H, W, C = rgb_out.shape
        # rgb_q ir_q是pool之后的B 7 7 C//2的特征
        W_q = self.conv(torch.cat([rgb_q, ir_q], dim=3).permute(0, 3, 1, 2).contiguous()) # B h w C --> B C h w --> BChw
        W_q = self.pool(W_q).view(B, C) # BC
        W_q = self.fc(W_q).softmax(dim=1) # B2

        W_qr = W_q[:, 0:1] 
        # print('W_qr', W_qr)
        W_qi = W_q[:, 1:2]
        # print('W_qi', W_qi)

        rgb_out = rgb_out.permute(0, 3, 1, 2).contiguous() # B C H W
        ir_out = ir_out.permute(0, 3, 1, 2).contiguous()

        rgb_out = rgb_out * W_qr.view(B,1,1,1)
        ir_out = ir_out * W_qi.view(B,1,1,1)

        return rgb_out, ir_out



class LFCA(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv = Conv(2*c1, 2)

    def forward(self, rgb, ir):
        fea = torch.cat([rgb, ir], dim=1)
        w = self.conv(fea).softmax(dim=1)
        w1 = w[:, 0:1, :, :]  # Assuming the first channel corresponds to w1
        # print('w1', w1.shape)
        w2 = w[:, 1:2, :, :]  # Assuming the second channel corresponds to w2
        # print('w2', w2.shape)
        rgb = rgb * w1
        ir = ir * w2
        return rgb, ir
   
    
class MyDecoup(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super().__init__()
        self.c1 = c1 # input_channel
        self.c2 = c2 # output_channel
        self.efd_ir = EFD(c1)
        self.efd_rgb = EFD(c1)
        self.avg = Avg()
        self.conv = Conv(3*c1, c2, 1, 1)
        self.conv2c_c = Conv(2*c1, c1, 1, 1)
        self.gfca_list = []
        self.gfca=GFCA(c1)
        self.lfca = LFCA(c1)
        self.conv_f = Conv(2*c1, c2)

    def forward(self, x):
        rgb_fea = x[0]   # dim:[b, c, h, w]
        ir_fea = x[1]   # dim:[b, c, h, w]
        b, c, h, w = rgb_fea.shape
        # 法一：subtraction
        # rgb_ir_diff = rgb_fea - ir_fea 
        # 法二：concate
        rgb_ir_diff = torch.cat([rgb_fea, ir_fea], dim=1)
        rgb_ir_diff = self.conv2c_c(rgb_ir_diff)
        # 法三：linaer_subtraction
        # rgb_ir_diff = self.rgb_mlp(rgb_fea) - self.ir_mlp(ir_fea)
        rgb_fea_com, rgb_fea_exc = self.efd_rgb(rgb_ir_diff, rgb_fea)  # dim:[b, c1, h, w]
        ir_fea_com, ir_fea_exc = self.efd_ir(rgb_ir_diff, ir_fea)     # dim:[b, c1, h, w]
        com_fea = self.avg(rgb_fea_com, ir_fea_com)  # dim:[b, c1, h, w]
        rgb_out, ir_out = self.gfca(rgb_fea_exc, ir_fea_exc)
        rgb, ir = self.lfca(rgb_out, ir_out)
        fea = torch.cat([com_fea, (rgb+ir)], dim=1)
        # print(fea.shape)
        out = self.conv_f(fea)
        return out, [rgb_fea_com, ir_fea_com, rgb_fea_exc, ir_fea_exc]