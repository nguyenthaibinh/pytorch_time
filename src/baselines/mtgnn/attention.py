import torch as th
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

class AttentionConvHead(nn.Module):
    def __init__(self, in_dim, d_k, d_v, t_kernel_size, t_stride=1, t_padding=0, t_dilation=1, side='both',
                 time_last=False, bias=True):
        super(AttentionConvHead, self).__init__()
        # t_padding = t_dilation * (t_kernel_size - 1) // 2
        self.d_v = d_v
        self.d_k = d_k
        self.t_kernel_size = t_kernel_size
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.t_dilation = t_dilation
        self.time_last = time_last
        self.padding = (t_padding, 0)
        self.side = side

        self.conv_q = nn.Conv2d(in_dim, d_k, kernel_size=1, bias=bias)
        self.conv_k = nn.Conv2d(in_dim, d_k, kernel_size=1, padding=self.padding, bias=bias)
        self.conv_v = nn.Conv2d(in_dim, d_v, kernel_size=1, padding=self.padding, bias=bias)

    def forward(self, q, k, v, mask=None):
        if self.time_last:
            q = q.permute(0, 1, 3, 2)
            k = k.permute(0, 1, 3, 2)
            v = v.permute(0, 1, 3, 2)
        B, C, T_in, N = q.size()           # batch_size, C_in, T_in, N
        # (w+2*pad-(d(k-1)+1))/s+1
        # T_out = (T_in + 2 * self.t_padding - self.t_dilation*(self.t_kernel_size - 1) - 1)/self.t_stride + 1

        K = self.conv_k(k)       # [B, C, T_in, N]
        V = self.conv_v(v)       # [B, C, T_in, N]

        K = F.unfold(K, kernel_size=(self.t_kernel_size, 1), stride=(self.t_stride, 1),
                     dilation=(self.t_dilation, 1))     # [B, C x t_kernel_size, L_T x N]
        V = F.unfold(V, kernel_size=(self.t_kernel_size, 1), stride=(self.t_stride, 1),
                     dilation=(self.t_dilation, 1))     # [B, C x t_kernel_size, L_T x N]

        K = K.contiguous().view(B, self.d_k, -1, N, self.t_kernel_size)  # [B, C, T_out, N, t_kernel_size]
        V = V.contiguous().view(B, self.d_v, -1, N, self.t_kernel_size)  # [B, C, T_out, N, t_kernel_size]

        T_out = V.size(2)
        if self.side == 'both':
            offset = int((T_in - T_out) / 2)
            Q = self.conv_q(q)[:, :, offset:T_in-offset, :]  # [B, C, T_out, N]

        elif self.side == 'left':
            l_offset = int(T_in - T_out)
            Q = self.conv_q(q)[:, :, l_offset:, :]  # [B, C, T_out, N]
        elif self.side == 'right':
            r_offset = int(T_in - T_out)
            Q = self.conv_q(q)[:, :, :-r_offset, :]  # [B, C, T_out, N]

        Q = F.unfold(Q, kernel_size=(1, 1), stride=(self.t_stride, 1))
        Q = Q.view(B, self.d_k, -1, N, 1)  # [B, C, T_out, N, 1]

        out = Q * K     # [B, C, T_out, N, t_kernel_size]
        out = F.softmax(out, dim=-1)    # [B, C, T_out, N, t_kernel_size]
        out = th.einsum('bchwk,bchwk -> bchw', out, V)   # [B, C, T_out, N]
        out = out.view(B, -1, T_out, N)     # [B, C, T_out, N]
        if self.time_last:
            out = out.permute(0, 1, 3, 2)
        return out

class MultiHeadAttentionConv(nn.Module):
    def __init__(self, in_dim, out_dim, t_kernels, t_strides, t_paddings, t_dilations, n_heads=None, side='both',
                 time_last=False, activation=th.relu):
        super(MultiHeadAttentionConv, self).__init__()
        self.activation = activation
        if n_heads is None:
            n_heads = len(t_kernels)
        d_v = out_dim // n_heads
        d_k = out_dim // n_heads
        assert out_dim == d_v * n_heads, f"Assertion Error: out_dim: {out_dim}, n_heads: {n_heads}, d_v: {d_v}."
        self.heads = nn.ModuleList([AttentionConvHead(in_dim, d_k, d_v, t_kernels[i], t_strides[i], t_paddings[i],
                                                      t_dilations[i], side=side, time_last=time_last)
                                    for i in range(n_heads)])

    def forward(self, q, k=None, v=None, mask=None):
        a = []
        min_len = 1000
        if k is None:
            k = q
        if v is None:
            v = q

        for head in self.heads:
            y = head(q, k, v, mask=mask)
            if y.size(2) < min_len:
                min_len = y.size(2)
            a.append(y)

        a = th.cat(a, dim=1)

        return a