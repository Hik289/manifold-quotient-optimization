"""Actual architectures for every named proposal family (no task aliases)."""
import math
import torch
from torch import nn
from geometry_torch import LowRankLinear

def replace_linear(old,rank):
    weight=old.weight.detach()
    out,ind=weight.shape
    new=LowRankLinear(ind,out,rank,bias=getattr(old,'bias',None) is not None).to(weight.device)
    with torch.no_grad():
        new.W0.copy_(weight)
        if new.bias is not None:new.bias.copy_(old.bias)
    return new

class CoordinateMLP(nn.Module):
    def __init__(self,ind,rank,kind='mlp',out=1):
        super().__init__();self.kind=kind
        if kind=='fourier':self.register_buffer('freq',torch.randn(ind,32)*4);ind=64
        self.first=nn.Linear(ind,128)
        self.hidden=nn.ModuleList([LowRankLinear(128,128,rank) for _ in range(2)])
        self.last=nn.Linear(128,out)
        if kind=='siren':
            with torch.no_grad():
                self.first.weight.uniform_(-1/ind,1/ind)
                for layer in self.hidden:layer.W0.uniform_(-math.sqrt(6/128)/30,math.sqrt(6/128)/30)
    def act(self,x):return torch.sin(30*x) if self.kind=='siren' else torch.tanh(x)
    def forward(self,x):
        if self.kind=='fourier':
            z=2*torch.pi*x@self.freq;x=torch.cat([z.sin(),z.cos()],-1)
        x=self.act(self.first(x))
        for layer in self.hidden:x=self.act(layer(x))
        return self.last(x)

def vision_model(name,classes,size,rank):
    if name=='mlp':return nn.Sequential(nn.Flatten(),nn.Linear(3*size*size,256),nn.ReLU(),LowRankLinear(256,256,rank),nn.ReLU(),LowRankLinear(256,256,rank),nn.ReLU(),nn.Linear(256,classes))
    if name=='resnet18':
        from torchvision.models import resnet18
        net=resnet18(weights=None)
        if size<=64:net.conv1=nn.Conv2d(3,64,3,1,1,bias=False);net.maxpool=nn.Identity()
        net.fc=nn.Sequential(nn.Linear(512,256),nn.ReLU(),LowRankLinear(256,256,rank),nn.ReLU(),nn.Linear(256,classes))
        return net
    import timm
    net=timm.create_model('vit_small_patch16_224',pretrained=False,num_classes=classes,img_size=size,patch_size=4 if size==32 else 16)
    for block in net.blocks:
        block.attn.qkv=replace_linear(block.attn.qkv,rank)
        block.attn.proj=replace_linear(block.attn.proj,rank)
        block.mlp.fc1=replace_linear(block.mlp.fc1,rank);block.mlp.fc2=replace_linear(block.mlp.fc2,rank)
    return net

class GraphNet(nn.Module):
    def __init__(self,ind,classes,c):
        super().__init__();kind=c['model'];rank=c['rank'];place=c['placement'];self.kind=kind
        self.input=nn.Linear(ind,128);self.output=nn.Linear(128,classes)
        if kind=='gcnii':
            self.convs=nn.ModuleList([nn.Linear(128,128,bias=False) for _ in range(8)])
        else:
            from torch_geometric.nn import GCNConv,GATConv,SAGEConv
            cls={'gcn':GCNConv,'gat':GATConv,'sage':SAGEConv}[kind]
            self.convs=nn.ModuleList([cls(128,128) for _ in range(3)])
        if place=='input':self.input=replace_linear(self.input,rank)
        elif place=='classifier':self.output=replace_linear(self.output,rank)
        else:
            for i,conv in enumerate(self.convs):
                if kind=='gcnii':self.convs[i]=replace_linear(conv,rank)
                else:
                    for attr in ('lin','lin_l','lin_r'):
                        lin=getattr(conv,attr,None)
                        if lin is not None:setattr(conv,attr,replace_linear(lin,rank))
    def forward(self,x,edge):
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        x=torch.relu(self.input(x));initial=x
        if self.kind=='gcnii':
            e,w=gcn_norm(edge,num_nodes=x.shape[0],add_self_loops=True)
            adj=torch.sparse_coo_tensor(e,w,(len(x),len(x))).coalesce()
        for i,conv in enumerate(self.convs):
            if self.kind=='gcnii':
                z=.9*torch.sparse.mm(adj,x)+.1*initial
                beta=math.log(.5/(i+1)+1);x=torch.relu((1-beta)*z+beta*conv(z))
            else:x=torch.relu(conv(x,edge))
        return self.output(x)

class DeepONet(nn.Module):
    def __init__(self,rank,n=32):
        super().__init__()
        self.branch=nn.Sequential(nn.Linear(n*n,128),nn.Tanh(),LowRankLinear(128,128,rank),nn.Tanh(),nn.Linear(128,128))
        self.trunk=CoordinateMLP(2,rank,out=128)
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,field,coords):return self.branch(field.flatten(1))@self.trunk(coords).T+self.bias

class SpectralConv(nn.Module):
    def __init__(self,width=64,modes=8):
        super().__init__();self.modes=modes
        self.weight=nn.Parameter(torch.randn(2,width,width,modes,modes,dtype=torch.cfloat)/(width*width))
    def forward(self,x):
        f=torch.fft.rfft2(x);out=torch.zeros_like(f);k=min(self.modes,f.shape[-2]//2,f.shape[-1])
        out[:,:,:k,:k]=torch.einsum('bixy,ioxy->boxy',f[:,:,:k,:k],self.weight[0,:,:,:k,:k])
        out[:,:,-k:,:k]=torch.einsum('bixy,ioxy->boxy',f[:,:,-k:,:k],self.weight[1,:,:,:k,:k])
        return torch.fft.irfft2(out,s=x.shape[-2:])

class FNO(nn.Module):
    def __init__(self,rank):
        super().__init__();self.lift=nn.Linear(3,64)
        self.spectral=nn.ModuleList([SpectralConv() for _ in range(4)])
        self.pointwise=nn.ModuleList([LowRankLinear(64,64,rank) for _ in range(4)])
        self.out=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,field,coords):
        n=field.shape[-1];xy=coords.reshape(n,n,2).expand(len(field),-1,-1,-1)
        x=self.lift(torch.cat([field[...,None],xy],-1))
        for spectral,linear in zip(self.spectral,self.pointwise):x=torch.nn.functional.gelu(spectral(x.permute(0,3,1,2)).permute(0,2,3,1)+linear(x))
        return self.out(x).squeeze(-1).flatten(1)

def language_model(name,rank,vocab):
    from transformers import GPT2Config,GPT2LMHeadModel
    width,layers,heads={'10m':(160,6,5),'25m':(320,7,5),'50m':(512,8,8)}[name]
    net=GPT2LMHeadModel(GPT2Config(vocab_size=vocab,n_embd=width,n_layer=layers,n_head=heads,n_positions=256,n_ctx=256,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.))
    # GPT-2 Conv1D stores transposed linear weights.
    for block in net.transformer.h:
        for parent,attr in [(block.attn,'c_attn'),(block.attn,'c_proj'),(block.mlp,'c_fc'),(block.mlp,'c_proj')]:
            old=getattr(parent,attr);new=LowRankLinear(old.weight.shape[0],old.weight.shape[1],rank)
            with torch.no_grad():new.W0.copy_(old.weight.T);new.bias.copy_(old.bias)
            setattr(parent,attr,new)
    return net
