from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile


class VISION(nn.Module): 
	def __init__(self,channel = 16):
		super(VISION,self).__init__()
		self.aoe = AOE(channel)
		self.gsao = GSAO(channel)

	def forward(self,x):
		x_aoe = self.aoe(x)
		out = self.gsao(x_aoe)

		return out
     
class GSAO(nn.Module): 
	def __init__(self,channel = 16):
		super(GSAO,self).__init__()
          
		self.gsao_left = GSAO_Left(channel)

		self.ssdc  = SSDC(channel)

		self.gsao_right = GSAO_Right(channel)
                    
		self.gsao_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)

	def forward(self,x):

		L,M,S,SS = self.gsao_left(x)
		ssdc = self.ssdc(SS)
		x_out = self.gsao_right(ssdc,SS,S,M,L)
		out = self.gsao_out(x_out)

		return out
     

class AOE(nn.Module):
	def __init__(self,channel = 16):
		super(AOE,self).__init__()
          
		self.uoa = UOA(channel)
		self.scp = SCP(channel)

	def forward(self,x):
		x_in = self.uoa(x)	
		x_out = self.scp(x_in)#3 16

		return x_out
     
class UOA(nn.Module): 
	def __init__(self,channel = 16):
		super(UOA,self).__init__()

		self.Haze_in1 = nn.Conv2d(1,channel,kernel_size=1,stride=1,padding=0,bias=False)
		self.Haze_in3 = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)  
		self.Haze_in4 = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)                             

	def forward(self,x):
		if x.shape[1] == 1:
			x_in = self.Haze_in1(x)#3 16
		elif x.shape[1] == 3:
			x_in = self.Haze_in3(x)#3 16
		elif x.shape[1] == 4:
			x_in = self.Haze_in4(x)#3 16
		
		return x_in
     
class SCP(nn.Module):
    def __init__(self, channel):
        super(SCP, self).__init__()
        self.cgm = CGM(channel)
        self.cim = CIM(channel)

    def forward(self, x):
        x_cgm = self.cgm(x)
        x_cim = self.cim(x_cgm, x)

        return x_cim

class GSAO_Left(nn.Module):
	def __init__(self,channel):
		super(GSAO_Left,self).__init__()    

		self.el = GARO(channel)#16
		self.em = GARO(channel*2)#32
		self.es = GARO(channel*4)#64
		self.ess = GARO(channel*8)#128
		self.esss = GARO(channel*16)#256
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#16 32
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#32 64
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 128
        
	def forward(self,x):
        
		elout = self.el(x)#16
		x_emin = self.conv_eltem(self.maxpool(elout))#32
		emout = self.em(x_emin)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin)
		x_esin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_esin)#128

		return elout,emout,esout,essout

class SSDC(nn.Module):
	def __init__(self,channel):
		super(SSDC,self).__init__()    

		self.s1 = SKO(channel*8)#128
		self.s2 = SKO(channel*8)#128

	def forward(self,x):
		ssdc1 = self.s1(x) + x
		ssdc2 = self.s2(ssdc1) + ssdc1

		return ssdc2

class GSAO_Right(nn.Module):
	def __init__(self,channel):
		super(GSAO_Right,self).__init__()    

		self.dss = GARO(channel*8)#128
		self.ds = GARO(channel*4)#64
		self.dm = GARO(channel*2)#32
		self.dl = GARO(channel)#16
        
		self.conv_dssstdss = nn.Conv2d(16*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#256 128
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 64
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 32
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)#32 16
        
	def _upsample(self,x):
		_,_,H,W = x.size()
		return F.upsample(x,size=(2*H,2*W),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):

		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout))
		dlout = self.dl(x_dlin+l)
        
		return dlout


class SKO(nn.Module):
    def __init__(self, in_ch, M=3, G=1, r=4, stride=1, L=32) -> None:
        super().__init__()
       
        d = max(int(in_ch/r), L) 
        self.M = M
        self.in_ch = in_ch
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3+i*2, stride=stride, padding = 1+i, groups=G),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
                )
            )
        # print("D:", d)
        self.fc = nn.Linear(in_ch, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, in_ch))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs): 
            fea = conv(x).clone().unsqueeze_(dim=1).clone() 
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas.clone(), fea], dim=1) 
        fea_U = torch.sum(feas.clone(), dim=1)  
        fea_s = fea_U.clone().mean(-1).mean(-1) 
        fea_z = self.fc(fea_s) 
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).clone().unsqueeze_(dim=1) 
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors.clone(), vector], dim=1)
        attention_vectors = self.softmax(attention_vectors.clone())
        attention_vectors = attention_vectors.clone().unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).clone().sum(dim=1)
        return fea_v


class GARO(nn.Module): 
	def __init__(self, channel, norm=False):
		super(GARO, self).__init__()

		self.conv_1_1 = DeformConv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)  
		self.conv_2_1 = DeformConv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False) 
		self.act = nn.PReLU(channel)  
		self.norm = nn.GroupNorm(num_channels=channel, num_groups=1) 

	def _upsample(self, x, y): 
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode='bilinear')

	def forward(self, x):
		x_1 = self.act(self.norm(self.conv_1_1(x)))
		x_2 = self.act(self.norm(self.conv_2_1(x_1))) + x
          
		return x_2

class CGM(nn.Module):
    def __init__(self, channel, prompt_len=3, prompt_size=96, lin_dim=16):
        super(CGM, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, channel, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class CIM(nn.Module):
    def __init__(self, channel):
        super(CIM, self).__init__()
        self.res = ResBlock(2*channel, 2*channel)
        self.conv3x3 = nn.Conv2d(2*channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, prompt, x):

        x = torch.cat((prompt, x), dim=1)
        x = self.res(x)
        out = self.conv3x3(x)

        return out


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
    

from thop import profile

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = VISION().to(device)

    input = torch.randn(1, 4, 512, 512).to(device)
    output = net(input)

    macs, params = profile(net, inputs=(input, ))
    
    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f G, params: %.2f M' % (macs / 1000000000.0, params / 1000000.0))
    print(output.shape)