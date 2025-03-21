# implement LoRA
import torch
from torch import nn
import math

class LoRA(nn.Module):
    """
    Low-Rank Adaptation
    """
    def __init__(self, in_features, out_features, rank=8, alpha=1, dropout=0.1, merge=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.merge = merge

        self.linear = nn.Linear(in_features, out_features)
        
        # Low-Rank Adaptation: W = W0 + AB, B∈R^(out_features×rank), A∈R^(rank×in_features)
        if rank > 0:
            self.A = nn.Parameter(torch.randn(out_features, rank))
            self.B = nn.Parameter(torch.randn(rank, in_features))
        
        # initialize
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        if merge:
            self.merge_weight()
       
    # merge weights
    def merge_weight(self):
        if self.merge and self.rank > 0:
            self.linear.weight.data = self.linear.weight.data + (self.A @ self.B) * self.scaling

    # unmerge weights
    def unmerge_weight(self):
        if self.merge and self.rank > 0:
            self.linear.weight.data = self.linear.weight.data - (self.A @ self.B) * self.scaling


    def forward(self, x):
        if self.rank > 0:
            # W = W0 + AB
            output_part1 = self.linear(x)
            output_part2 = self.scaling * (x @ (self.A @ self.B).T)
            output = output_part1 + output_part2
        else:
            output = self.linear(x)

        output = self.dropout(output)

        return output


batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
lora_alpha = 16
dropout = 0.1

# Create a test input
x = torch.randn(batch_size, seq_len, in_features)

# Test regular mode (no merge)
lora_layer = LoRA(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    alpha=lora_alpha,
    dropout=dropout,
    merge=False
)

# Forward pass
output = lora_layer(x)
print(f"Output shape (no merge): {output.shape}")  # Should be [batch_size, seq_len, out_features]

# Test merged mode
lora_layer_merged = LoRA(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    alpha=lora_alpha,
    dropout=dropout,
    merge=True
)

# Forward pass with merged weights
output_merged = lora_layer_merged(x)
print(f"Output shape (merged): {output_merged.shape}")  # Should be [batch_size, seq_len, out_features]

# Test weight merging/unmerging
lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

print("Max difference after merge/unmerge cycle:", 
      torch.max(torch.abs(output - output_after_unmerge)).item())
