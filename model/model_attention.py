import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.2, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)   # [patch_size, D]
        b = self.attention_b(x)   # [patch_size, D]
        A = a.mul(b)              # [patch_size, D]
        A = self.attention_c(A)   # [patch_size, n_classes]
        return A, x


class MIL_attn(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.2, n_classes=1, input_dim=2048, attention_only=False):
        super(MIL_attn, self).__init__()
        self.attention_only = attention_only
        self.n_classes = n_classes
        self.fc_1 = nn.Sequential(*[nn.Linear(input_dim, L), nn.ReLU(), nn.Dropout(dropout)])
        self.attn_net = Attn_Net_Gated(L=L, D=D, dropout=dropout, n_classes=n_classes)
        self.fc_2 = nn.Sequential(*[nn.Linear(L, D), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = nn.Linear(D, n_classes)
        
    def forward(self, x):
        x = self.fc_1(x)           # x: [patch_size, input_dim] --> [patch_size, L]
        A, h = self.attn_net(x)    # A: [patch_size, n_classes], h: [patch_size, L]
        A = torch.transpose(A, 1, 0)  # A: [n_classes, patch_size]
        if self.attention_only:
            return A
        A = torch.mm(F.softmax(A, dim=1), h)     # A: [n_classes, L]  # weighted sum of patches using attention weights
        A = self.fc_2(A)       # A: [n_classes, L] --> [n_classes, D]
        logits = self.classifier(A)  # logits: [n_classes, D] --> [n_classes, n_classes]
        return logits

