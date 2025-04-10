
def _beta_schedule(t, T, beta_1, beta_T, kind):
    if (kind == 'linear'):
        beta = beta_1 + (beta_T - beta_1)*(t-1)/(T-1)
    if (kind == 'quadratic'):
        beta = beta_1 + (beta_T - beta_1)*((t-1)/(T-1))**2
    return beta

def _alpha_schedule(t, T, beta_1, beta_T, kind):
    alpha = 1 - _beta_schedule(t, T, beta_1, beta_T, kind)
    return alpha

def _alpha_bar_schedule(t, T, beta_1, beta_T, kind):
    alpha_bar = 1
    for i in range(1, t+1):
        alpha_bar *= _alpha_schedule(i, T, beta_1, beta_T, kind)
    return alpha_bar

class beta_scheduler():
    def __init__(self, beta_min, beta_max, kind = 'linear'):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.kind = kind
    

    def beta_schedule(self, T):
        return torch.linspace(self.beta_min, self.beta_max, T).float()
    
    def alpha_schedule(self, T):
        return torch.linspace(self)
    def beta_schedule(self, T):
        return [_beta_schedule(t, T, self.beta_min, self.beta_max, self.kind) for t in range(1, T+1)]
    
    def alpha_schedule(self, T):
        return [_alpha_schedule(t, T, self.beta_min, self.beta_max, self.kind) for t in range(1, T+1)]
    
    def alpha_bar_schedule(self, T):
        return [_alpha_bar_schedule(t, T, self.beta_min, self.beta_max, self.kind) for t in range(1, T+1)]