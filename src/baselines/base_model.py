from torch import nn

class BaseModel(nn.Module):
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def get_init_adj(self):
        return None

    def get_static_adj(self):
        return None

    def get_dynamic_adj(self):
        return None