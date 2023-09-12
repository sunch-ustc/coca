import torch
import torch.nn as nn
from torch import einsum, nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from monai.transforms import Resize,Compose, RandGaussianNoised,Resized
from torchvision.utils import save_image
from einops import rearrange, repeat
from typing import List, Union, Optional
import pdb
import copy
from sklearn.manifold import TSNE
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class ResNet(nn.Module):
    def __init__(self,
                 block: Union[BasicBlock, Bottleneck],
                 layers: List[int],
                 cfg,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[nn.Module] = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.cfg = copy.deepcopy(cfg)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups 
        self.base_width = width_per_group 
        if self.cfg.rgb_trans == True: 
            self.rgb_trans = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d( self.inplanes, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
            
        ).cude()
        
        self.conv1 = nn.Conv2d(in_channels,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        dim_list = { "layer1":56*56, "layer2": 28 * 28, "layer3": 14 * 14 ,"layer4": 7 * 7  }
        embed_dim_ratio = 1
        num_heads = 2 
        if "heads4" in self.cfg.subnet : num_heads = 4
        elif "heads8" in self.cfg.subnet : num_heads = 8
        elif "heads16" in self.cfg.subnet : num_heads = 16
        
        if "all" in self.cfg.subnet: attention_dim = 14 * 14
        if "layer1" in self.cfg.subnet:  attention_dim = 56*56
        elif  "layer2" in self.cfg.subnet: attention_dim = 28 * 28
        elif  "layer3" in self.cfg.subnet: attention_dim = 14 * 14
        elif  "layer4" in self.cfg.subnet: 
            attention_dim = 7 * 7
            embed_dim_ratio = 6 / 7
        self.BIAttention1 = BiAttentionBlock(cfg=cfg, v_dim = attention_dim, l_dim = attention_dim , embed_dim = int(attention_dim * embed_dim_ratio), num_heads = num_heads ) if "sub_mod_attention" in self.cfg.subnet else None
        self.BIAttention2 = BiAttentionBlock(cfg=cfg, v_dim = attention_dim, l_dim = attention_dim , embed_dim = int(attention_dim * embed_dim_ratio), num_heads = num_heads ) if "sub_mod_attention" in self.cfg.subnet else None
        self.BIAttention3 = BiAttentionBlock(cfg=cfg, v_dim = attention_dim, l_dim = attention_dim , embed_dim = int(attention_dim * embed_dim_ratio), num_heads = num_heads ) if "sub_mod_attention" in self.cfg.subnet else None
        
        sub_layer0 = self.create_layer(in_channels ,norm_layer,block=block, layers=layers,location = "sub_layer0" )
        self.layer0 = sub_layer0[0] 
        self.layer0_sub0 = sub_layer0[0] if "sub_same_mod" not in self.cfg.subnet else 0
        self.layer0_sub1 = sub_layer0[1] if "sub_same_mod" not in self.cfg.subnet else 0
        self.layer0_sub2 = sub_layer0[2] if "sub_same_mod" not in self.cfg.subnet else 0
        self.layer1 = self._make_layer(block, 64, layers[0])
        sub_layer1 = self.create_layer(in_channels ,norm_layer,block=block, layers=layers,location = "sub_layer1" )
        self.layer1_sub0 = sub_layer1[0]
        self.layer1_sub1 = sub_layer1[1]
        self.layer1_sub2 = sub_layer1[2] 
        concat_num = 3  
        if "layer1" in self.cfg.subnet: feature_dim_mod = 64
        elif "layer2" in self.cfg.subnet: feature_dim_mod = 128
        else: feature_dim_mod = 64
            
        self.conv1x1 = nn.Conv2d(feature_dim_mod*concat_num, feature_dim_mod, kernel_size=1, stride=1, padding=0, bias=False) if "concat" in self.cfg.subnet else  None
        
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        sub_layer2 = self.create_layer(in_channels ,norm_layer, block=block, layers=layers,location = "sub_layer2"
                                       ,replace_stride_with_dilation=replace_stride_with_dilation )
        self.layer2_sub0 = sub_layer2[0]
        self.layer2_sub1 = sub_layer2[1]
        self.layer2_sub2 = sub_layer2[2]  
        self.layer2_relu = nn.ReLU(inplace=True)
        self.layer2_bn1 = norm_layer(self.inplanes) if "bn_relu" in self.cfg.subnet else 0
        self.layer2_bn2 = norm_layer(self.inplanes) if "bn_relu" in self.cfg.subnet else 0
        self.layer2_bn3 = norm_layer(self.inplanes) if "bn_relu" in self.cfg.subnet else 0                        
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        sub_layer3 = self.create_layer(in_channels ,norm_layer, block=block, layers=layers,location = "sub_layer3"
                                       ,replace_stride_with_dilation=replace_stride_with_dilation )
        self.layer3_sub0 = sub_layer3[0]
        self.layer3_sub1 = sub_layer3[1]
        self.layer3_sub2 = sub_layer3[2] 
        
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2] )
        self.conv1x1_4 = nn.Conv2d(512* self.cfg.cut_region , 512, kernel_size=1, stride=1, padding=0, bias=False) if "concat" in self.cfg.subnet else  None
        self.conv1x1_frame = nn.Conv2d(512* 8 , 512, kernel_size=1, stride=1,
                                       padding=0, bias=False) if "concat" in self.cfg.outpooling else  None
        self.conv1x1_mod = nn.Conv2d(512* 3 , 512, kernel_size=1, stride=1,
                                       padding=0, bias=False) if "layer4cat" in self.cfg.subnet else  None
        sub_layer4 = self.create_layer(in_channels ,norm_layer, block=block, layers=layers,location = "sub_layer4"
                                       ,num_subnet = 5,replace_stride_with_dilation=replace_stride_with_dilation )
        self.layer4_sub0 = sub_layer4[0]
        self.layer4_sub1 = sub_layer4[1]
        self.layer4_sub2 = sub_layer4[2] 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        '''Cross attention '''   
         
        num_model = 1
        if "sub_mod" in self.cfg.subnet   : 
            num_model = 3
        num_img_queries = 0 
        input_dim = 512 * num_model       # 输入的维度 
        dim_head = 256 * num_model  # key value 的linear的输出维度,也是每一个 head 的维度
        heads = int( input_dim /  dim_head   )    # head 的数量
        # 输出的维度 = heads * x_dim （ * 2）      “*2”代表进行skip操作
        self.img_attn_pool = CrossAttention(input_dim=input_dim,  dim_head=dim_head, heads=heads, norm_x=True,cfg = self.cfg) if cfg.outpooling=="attention" else None
        self.img_queries = nn.Parameter(torch.ones(num_img_queries + 1, dim_head * heads)) # num image queries for multimodal, but 1 extra CLS for contrastive learning 
         
        fc_input =  512 * block.expansion
        if self.cfg.region_skip == 0  : 
            fc_input = 512 * block.expansion *  self.cfg.cut_region  
        if ( "concat" in self.cfg.subnet  or  cfg.outpooling=="concat" )and "mod_region" not in self.cfg.subnet :  fc_input = 512 * block.expansion
        if  "sub_mod_attention" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet or "sub_same_mod" in self.cfg.subnet:
            fc_input = 512 * block.expansion * 3 
        if self.cfg.outpooling== "attention":
            fc_input = fc_input * heads 
        self.fc =  nn.Sequential( nn.Linear(fc_input  , 512) ,
                nn.ReLU(inplace=True),
                nn.Linear(512  , num_classes) 
                )
        self.fc1 = nn.Linear(fc_input  , 512)
        self.fc2 =  nn.Sequential(  nn.ReLU(inplace=True),
                nn.Linear(512  , num_classes) 
                )
        # self.fc1_1 =  nn.Linear(fc_input  , num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 
    def create_layer(self,in_channels ,norm_layer, num_subnet = 3, **kwargs):
        '''
        self.cfg.subnet = '' 输出的为参数, 减少对网络容量的占用
        '''
        layer_tp = []   
        for i in range( num_subnet ):
            if ("sub_layer0" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet or "mod_region" in   self.cfg.subnet 
                or "sub_mod_all_layer" in self.cfg.subnet or "sub_same_mod" in self.cfg.subnet) and kwargs["location"] == "sub_layer0" : 
                layer_tp.append(nn.Sequential( nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(self.inplanes), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1) ,
                ))
            elif ("sub_layer1" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet or "mod_region" in self.cfg.subnet 
                  or "sub_layer0" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet)  and kwargs["location"] == "sub_layer1"  : 
                layer_tp.append(self._make_layer(kwargs["block"], 64, kwargs["layers"][0]))
            elif ("sub_layer2" in self.cfg.subnet  or "sub_mod_attention" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet) and kwargs["location"] == "sub_layer2":
                self.inplanes = 64
                layer_tp.append(self._make_layer(kwargs["block"],
                                       128,
                                       kwargs["layers"][1],
                                       stride=2,
                                       dilate=kwargs["replace_stride_with_dilation"][0]))
            elif ("sub_layer3" in self.cfg.subnet  or "sub_mod_attention" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet) and kwargs["location"] == "sub_layer3" :
                self.inplanes = 128
                layer_tp.append(self._make_layer(kwargs["block"],
                                       256,
                                       kwargs["layers"][2],
                                       stride=2,
                                       dilate=kwargs["replace_stride_with_dilation"][1]))
            elif ("sub_layer4" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet or "mod_region" in   self.cfg.subnet 
                  or "sub_mod_all_layer" in self.cfg.subnet)  and kwargs["location"] == "sub_layer4" :
                self.inplanes = 256
                layer_tp.append(self._make_layer(kwargs["block"],
                                       512,
                                       kwargs["layers"][3],
                                       stride=2,
                                       dilate=kwargs["replace_stride_with_dilation"][2]))
                
            else:
                layer_tp.append(i)  
        return layer_tp
    
    def deepcopy(self):
        '''
        该函数用于 将加载的 resnet18 的参数, 加载到subnet中
        '''        
        #
        if "sub_same_mod" in self.cfg.subnet :
            layer_tp = nn.Sequential(self.conv1,self.bn1,self.relu ,self.maxpool) #,self.layer1
            self.layer0.load_state_dict(layer_tp.state_dict()) 
        if "sub_layer0" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet  or "sub_mod_attention" in self.cfg.subnet or "mod_region" in self.cfg.subnet :
            layer_tp = nn.Sequential(self.conv1,self.bn1,self.relu ,self.maxpool) #,self.layer1
            self.layer0_sub0.load_state_dict(layer_tp.state_dict())
            self.layer0_sub1.load_state_dict(layer_tp.state_dict())
            self.layer0_sub2.load_state_dict(layer_tp.state_dict()) 
        if  "sub_layer1" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet or "mod_region" in self.cfg.subnet  or "sub_layer0" in self.cfg.subnet   :
            self.layer1_sub0.load_state_dict(self.layer1.state_dict())
            self.layer1_sub1.load_state_dict(self.layer1.state_dict())
            self.layer1_sub2.load_state_dict(self.layer1.state_dict()) 
        if "sub_layer2" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet :
            self.layer2_sub0.load_state_dict(self.layer2.state_dict())
            self.layer2_sub1.load_state_dict(self.layer2.state_dict())
            self.layer2_sub2.load_state_dict(self.layer2.state_dict()) 
        if "sub_layer3" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet:
            self.layer3_sub0.load_state_dict(self.layer3.state_dict())
            self.layer3_sub1.load_state_dict(self.layer3.state_dict())
            self.layer3_sub2.load_state_dict(self.layer3.state_dict()) 
        if  "sub_layer4" in self.cfg.subnet or "sub_mod_all_layer" in self.cfg.subnet or "sub_mod_attention" in self.cfg.subnet or "mod_region" in   self.cfg.subnet :
            self.layer4_sub0.load_state_dict(self.layer4.state_dict())
            self.layer4_sub1.load_state_dict(self.layer4.state_dict())
            self.layer4_sub2.load_state_dict(self.layer4.state_dict()) 
            # pdb.set_trace()
        else:
            pass
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,tp=0):
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if tp == 1:
            pdb.set_trace()
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)
 
        
    def forward(self, x,  test = None, tsne_path = None, orth = None, **kwargs  ): 
        #pdb.set_trace() 
        reshape = False  
        B, C, T, H, W = x.shape
        
        if x.dim() == 5:
            reshape = True 
            B, C, T, H, W = x.shape
            x0 = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            x = cut_region_select(x0, shape = x.shape, cut_region = self.cfg.cut_region)
        if "sub_mod" in self.cfg.subnet : 
            try: 
                adv = kwargs['adv']   
                return self.sub_net(x,B=B, T=T, orth=orth,adv = True)
            except: 
                return self.sub_net(x,B=B, T=T, orth=orth)
        if self.cfg.subnet== "sub_same_mod"  : 
            return self.sub_net(x,B=B, T=T, orth=orth)
            x = x.view(B * T * C, H, W).unsqueeze(1).repeat(1,3,1,1)
        if test == True:
            pdb.set_trace()       
        if "sub_layer0" in self.cfg.subnet  or "mod_region"  in self.cfg.subnet:
            x = self.sub_net(x,B=B, T=T,location = 1)
        else:
            # if self.cfg.rgb_trans == True: 
            #     x = self.conv1(self.rgb_trans(x) + x)   
            # else:
            #     x = self.conv1(x)
            
            x = self.conv1(x) 
            #print("conv1:",x.sum())
            x = self.bn1(x) 
            #print("bn1:",x.sum())
            x = self.relu(x) 
            #print("relu:",x.sum())
            x = self.maxpool(x)  
            #print("maxpool:",x.sum())
        x = self.layer1(x) 
        #print("layer1:",x.sum())
        x = self.layer2(x) 
        #print("layer2:",x.sum())
        x = self.layer3(x)   
        #print("layer3:",x.sum())
        x = self.layer4(x)   
        #print("layer4:",x.sum())
            
        if self.cfg.outpooling== "concat": 
            num, channel,width,hight = x.shape
            x = x.view(B,  T *  self.cfg.cut_region*channel,width,hight)
            x = self.conv1x1_frame(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            output =  self.fc(x)
            return output
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        
        
        attention_scores = torch.tensor(0).cuda()
 
        x = torch.mean(x,dim=1)  
        x_avg = x
        x = self.fc1(x) 
        try: 
            adv = kwargs['adv']   
            if adv == "ce_ce":
                attention_scores = self.fc1_1(x_avg)
            else:
                # pdb.set_trace()
                attention_scores = x
        except: pass
            
        output = self.fc2(x) 
        return output, attention_scores 
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
def default(val, d):
    return val if exists(val) else d
def exists(val):
    return val is not None
# normalization
# they use layernorm without bias, something that pytorch does not offer 
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
# import torchvision.models as models 
# models.vgg
class CrossAttention(nn.Module):
    # From Coca
    def __init__(
        self,
        input_dim,
        *, 
        dim_head=128,
        heads=8,
        parallel_ff=False,
        ff_mult=1,
        norm_x=False,
        cfg = None
    ):
        super().__init__()
          
        self.cfg = cfg
        # self.fc = nn.Linear(input_dim   , x_dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head 

        self.norm = LayerNorm(input_dim)
        self.x_norm = LayerNorm(input_dim) if norm_x else nn.Identity()  

        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_k_global = nn.Linear(input_dim, dim_head, bias=False)
        self.to_kv = nn.Linear(input_dim, dim_head * 2, bias=False)
        inner_dim_out = inner_dim   
        if  self.cfg.region_skip == 1 or self.cfg.region_skip == 2  :
            inner_dim_out = inner_dim + input_dim 
        #self.to_out = nn.Linear(inner_dim_out, input_dim, bias=False)

        # whether to have parallel feedforward
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #ff_inner_dim = ff_mult * input_dim 
        # self.ff = nn.Sequential(
        #     nn.Linear(dim, ff_inner_dim * 2, bias=False),
        #     SwiGLU(),
        #     nn.Linear(ff_inner_dim, dim, bias=False)
        # ) if parallel_ff else None

    def forward(self, img_queries, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # Dimensionality reduction
        B, T, F = x.shape
        # x = x.contiguous().view(B * T, F)
        # x = x.view(B, T,  -1) 
        x0 = x
        # x = self.fc(x)
        # x = self.fc(x)
        # pre-layernorm, for queries and x
        img_queries = self.norm(img_queries) if self.cfg.mark != 'no-layernorm' else img_queries
        x = self.x_norm(x)   if self.cfg.mark != 'no-layernorm' else x
        # get queries
        q = img_queries
        # q = self.to_q(img_queries)
        # k_global = self.to_k_global(img_queries) 
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)  
        
        # get key / values

        k, v = self.to_kv(x).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q * self.scale, k) # scale #q = q * self.scale 

        # attention 
        
        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b j d -> b h i d', attn, x)
        out = rearrange(out, 'b h n d -> b  (n h d)') 
        
        # out = self.to_out(out)
        
        # add parallel feedforward (for multimodal layers) 
            
        # if exists(self.ff):
        #     out = out + self.ff(img_queries)

        return out,attn

def resnet18(cfg):
    return ResNet(BasicBlock, [2, 2, 2, 2],cfg=cfg,
                  in_channels=cfg.IN_CHANNELS,
                  num_classes=cfg.NUM_CLASSES)

def resnet34(cfg):
    return ResNet(BasicBlock, [3, 4, 6, 3],cfg=cfg,
                  in_channels=cfg.IN_CHANNELS,
                  num_classes=cfg.NUM_CLASSES)

def resnet50(cfg):
    return ResNet(Bottleneck, [3, 4, 6, 3],cfg=cfg,
                  in_channels=cfg.IN_CHANNELS,
                  num_classes=cfg.NUM_CLASSES)

def resnet101(cfg):
    return ResNet(Bottleneck, [3, 4, 23, 3],cfg=cfg,
                  in_channels=cfg.IN_CHANNELS,
                  num_classes=cfg.NUM_CLASSES)

def resnet152(cfg):
    return ResNet(Bottleneck, [3, 8, 36, 3],cfg=cfg,
                  in_channels=cfg.IN_CHANNELS,
                  num_classes=cfg.NUM_CLASSES)
def cut_region_select(x0, shape, cut_region = 0):
    reshape = True 
    B, C, T, H, W = shape 

    #x_ = x0.unsqueeze(1)  B * T, 1, C, H, W 
    # save_image(x0[:,:,H_quarter:H_half+H_quarter, W_quarter:W_half][0][5]  , "/code/tumor/fig/cut_center.png")
    # pdb.set_trace() 
    if cut_region == 5: 
        resize_monai=Resized(keys=["image" ],spatial_size=(3,224,224),allow_missing_keys=True) 
        H_half, W_half = int(H/2), int(W/2)
        x1 = resize_monai({'image':x0[:,:,0:H_half,  0:W_half].cpu()})['image'].cuda().unsqueeze(1)
        x2 = resize_monai({'image':x0[:,:,H_half:,   0: W_half].cpu()})['image'].cuda().unsqueeze(1)
        x3 = resize_monai({'image':x0[:,:,0:H_half,  W_half: ].cpu()})['image'].cuda().unsqueeze(1)
        x4 = resize_monai({'image':x0[:,:,H_half:,   W_half: ].cpu()})['image'].cuda().unsqueeze(1)
        list_x = [x0.unsqueeze(1),x1,x2,x3,x4]
        L = len(list_x)
        x = torch.cat(list_x,dim = 1 ).view(B * T * L, C, H, W)
    if cut_region == 4: 
        resize_monai=Resized(keys=["image" ],spatial_size=(3,224,224),allow_missing_keys=True) 
        H_half, W_half = int(H/2), int(W/2)
        x1 = resize_monai({'image':x0[:,:,0:H_half,  0:W_half].cpu()})['image'].cuda().unsqueeze(1)
        x2 = resize_monai({'image':x0[:,:,H_half:,   0: W_half].cpu()})['image'].cuda().unsqueeze(1)
        x3 = resize_monai({'image':x0[:,:,0:H_half,  W_half: ].cpu()})['image'].cuda().unsqueeze(1)
        x4 = resize_monai({'image':x0[:,:,H_half:,   W_half: ].cpu()})['image'].cuda().unsqueeze(1)
        list_x = [x1,x2,x3,x4]
        L = len(list_x)
        x = torch.cat(list_x,dim = 1 ).view(B * T * L, C, H, W)
    elif cut_region == 10: 
        resize_monai=Resized(keys=["image" ],spatial_size=(3,224,224),allow_missing_keys=True) 
        H_third, W_third = int(H/3), int(W/3)
        x1 = resize_monai({'image':x0[:,:,0:H_third,  0:W_third].cpu()})['image'].cuda().unsqueeze(1)
        x2 = resize_monai({'image':x0[:,:,0:H_third,  W_third: W_third*2].cpu()})['image'].cuda().unsqueeze(1)
        x3 = resize_monai({'image':x0[:,:,0:H_third,  W_third*2:  ].cpu()})['image'].cuda().unsqueeze(1)
        x4 = resize_monai({'image':x0[:,:,H_third:H_third*2,   0:W_third ].cpu()})['image'].cuda().unsqueeze(1)
        x5 = resize_monai({'image':x0[:,:,H_third:H_third*2,  W_third: W_third*2].cpu()})['image'].cuda().unsqueeze(1)
        x6 = resize_monai({'image':x0[:,:,H_third:H_third*2,  W_third*2: ].cpu()})['image'].cuda().unsqueeze(1)
        x7 = resize_monai({'image':x0[:,:,H_third*2: ,  0:W_third ].cpu()})['image'].cuda().unsqueeze(1)
        x8 = resize_monai({'image':x0[:,:,H_third*2:,  W_third: W_third*2 ].cpu()})['image'].cuda().unsqueeze(1)
        x9 = resize_monai({'image':x0[:,:,H_third*2:,    W_third*2: ].cpu()})['image'].cuda().unsqueeze(1)
        list_x = [x0.unsqueeze(1),x1,x2,x3,x4,x5,x6,x7,x8,x9]
        L = len(list_x)
        #pdb.set_trace()
        x = torch.cat(list_x,dim = 1 ).view(B * T * L, C, H, W)
    elif cut_region == 3:  
        x_T1 =  x0[:,0:1,:,:].repeat(1,3,1,1 ) 
        x_T1E =  x0[:,1:2,:,: ].repeat(1,3,1,1 ) 
        x_T2 =  x0[:,2:3,:,: ].repeat(1,3,1,1 )  
        # resize_monai=Resized(keys=["image" ],spatial_size=(3,224,224),allow_missing_keys=True) 
        # H_half, W_half = int(H/2), int(W/2)
        # x1 = resize_monai({'image':x0[:,:,0:H_half,  :].cpu()})['image'].cuda().unsqueeze(1)
        # x2 = resize_monai({'image':x0[:,:,H_half:,   :].cpu()})['image'].cuda().unsqueeze(1)
        
        list_x = [x_T1.unsqueeze(1),x_T1E.unsqueeze(1),x_T2.unsqueeze(1) ]
        L = len(list_x)
        x = torch.cat(list_x,dim = 1 ).view(B * T * L, C, H, W) 
        pass
    elif cut_region == 2:

        resize_monai=Resized(keys=["image" ],spatial_size=(3,224,224),allow_missing_keys=True) 
        H_half, W_half, H_quarter, W_quarter, H_eighth = int(H/2), int(W/2),int(H/4), int(W/4), int(H/8)
        x1 = resize_monai({'image':x0[:,:,H-H_half-H_eighth:H-H_eighth, W_quarter:W_quarter+ W_half].cpu()})['image'].cuda().unsqueeze(1) 
        
        list_x = [x0.unsqueeze(1),x1  ]
        L = len(list_x)
        x = torch.cat(list_x,dim = 1 ).view(B * T * L, C, H, W) 
        pass
    elif cut_region == 1:
        x = x0
        pass
    return x

class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True) if "no_lamuda" not in cfg.subnet else 1
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True) if "no_lamuda" not in cfg.subnet else 1

    def forward(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        # v = v + self.drop_path(self.gamma_v * delta_v)
        # l = l + self.drop_path(self.gamma_l * delta_l)
        l_fusion = self.drop_path(self.gamma_l * delta_l)
        v_fusion = self.drop_path(self.gamma_v * delta_v)
        return v_fusion, l_fusion
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True #cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = True #cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = True #cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l
if __name__ == '__main__':
    class cfg0():
        def __init__(self) -> None:
            self.IN_CHANNELS=3
            self.NUM_CLASSES=2
            self.outpooling='attention'
    cfg=cfg0()
    image=torch.randint(0,10,(1, 3, 60, 224, 224)).float() #218, 182
    model=resnet18(cfg)#build_model("RN50","/data/sunch/output/RN50.pt",device="cuda",cfg=cfg).cuda()
    output=model(image)
    for name,para in model.named_parameters():
 
        if name=="fc.weight" :
            break 
        para.requires_grad = False
 
       
    print(output)