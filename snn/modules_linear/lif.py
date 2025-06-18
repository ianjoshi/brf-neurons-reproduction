import torch
from .. import functional

################################################################
# Neuron update functional
################################################################


DEFAULT_LI_TAU_M = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_STD = 5.

DEFAULT_LIF_TAU_M = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_STD = 5.


def li_update(
        x: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor
) -> torch.Tensor:

    u = u.mul(alpha) + x.mul(1.0 - alpha)

    return u


def lif_update(
        x: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor,
        theta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    z = functional.StepLinearGrad.apply(u - theta)

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta)

    return z, u

################################################################
# Layer classes
################################################################

class LICell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            tau_mem: float = DEFAULT_LI_TAU_M,
            adaptive_tau_mem: bool = True,
            adaptive_tau_mem_mean: float = DEFAULT_LI_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_LI_ADAPTIVE_TAU_M_STD,
            bias: bool = False,
            use_linear_decay: bool = True  # [MOD] Added toggle for linear vs. exponential decay
    ) -> None:
        super(LICell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.bias = bias
        self.use_linear_decay = use_linear_decay  # [MOD] Store toggle in instance

        self.linear = torch.nn.Linear(input_size, layer_size, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight)

        if bias:
            torch.nn.init.constant_(self.linear.bias, 0)

        tau_mem = tau_mem * torch.ones(layer_size)

        if adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=adaptive_tau_mem_mean, std=adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tau_mem)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        in_sum = self.linear(x)
        tau_mem = torch.abs(self.tau_mem)

        if self.use_linear_decay:  # [MOD]
            alpha = 1.0 - (1.0 / tau_mem)  # [MOD]
            alpha = torch.clamp(alpha, 0.0, 1.0)  # [MOD]
        else:
            alpha = torch.exp(-1.0 / tau_mem)  # [MOD]

        u = u * alpha + in_sum * (1.0 - alpha)
        return u



class LIFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            tau_mem: float = DEFAULT_LIF_TAU_M,
            adaptive_tau_mem: bool = True,
            adaptive_tau_mem_mean: float = DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_LIF_ADAPTIVE_TAU_M_STD,
            bias: bool = False,
            use_linear_decay: bool = True  # [MOD] Added toggle for linear vs. exponential decay
    ) -> None:
        super(LIFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size
        self.use_linear_decay = use_linear_decay  # [MOD]

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        self.adaptive_tau_mem = adaptive_tau_mem
        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std

        tau_mem = tau_mem * torch.ones(layer_size)

        if adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=adaptive_tau_mem_mean, std=adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tensor=tau_mem)

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        z, u = state

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)

        if self.use_linear_decay:  # [MOD]
            alpha = 1.0 - (1.0 / tau_mem)  # [MOD]
            alpha = torch.clamp(alpha, 0.0, 1.0)  # [MOD]
        else:
            alpha = torch.exp(-1.0 / tau_mem)  # [MOD]

        z, u = lif_update(x=in_sum, u=u, alpha=alpha)

        return z, u



class LICellSigmoid(LICell):
    def __init__(self, *args, use_sigmoid_decay: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sigmoid_decay = use_sigmoid_decay  # [MOD]

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        in_sum = self.linear(x)

        if self.use_sigmoid_decay:  # [MOD]
            alpha = torch.sigmoid(self.tau_mem)
        elif self.use_linear_decay:
            tau_mem = torch.abs(self.tau_mem)
            alpha = 1.0 - (1.0 / tau_mem)
            alpha = torch.clamp(alpha, 0.0, 1.0)
        else:
            tau_mem = torch.abs(self.tau_mem)
            alpha = torch.exp(-1.0 / tau_mem)

        u = u * alpha + in_sum * (1.0 - alpha)
        return u




# Output LI with flexible bit precision
class LICellBP(LICell):
    def __init__(self,
                 *args,
                 bit_precision: int = 32,
                 use_linear_decay: bool = True,  # [MOD] New toggle
                 **kwargs
                 ):
        super().__init__(*args, use_linear_decay=use_linear_decay, **kwargs)  # [MOD]

        self.bit_precision = bit_precision
        self.use_linear_decay = use_linear_decay  # [MOD]

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        in_sum = self.linear(x)
        tau_mem = torch.abs(self.tau_mem)

        if self.use_linear_decay:  # [MOD]
            alpha = 1.0 - (1.0 / tau_mem)
            alpha = torch.clamp(alpha, 0.0, 1.0)
        else:
            alpha = torch.exp(-1.0 / tau_mem)

        alpha = functional.quantize_tensor(alpha, self.bit_precision)

        u = u * alpha + in_sum * (1.0 - alpha)
        return u



