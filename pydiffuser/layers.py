import torch


def linear(
    weight: torch.Tensor, bias: torch.Tensor, input: torch.Tensor
) -> torch.Tensor:
    """Applies a linear transformation to the incoming data. The weight must be
    a 2D tensor of dimensions (output_dim, input_dim), and the bias must be a 1D
    tensor of dimensions (output_dim)."""

    layer = torch.nn.Linear(weight.shape[0], weight.shape[1], device="cpu")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)


def group_norm(
    weight: torch.Tensor, bias: torch.Tensor, input: torch.Tensor, groups: int = None
) -> torch.Tensor:
    """Applies a group normalization to the incoming data. Each vector in the
    final layer is broken up into groups (by default the entire vector is a
    single group) and each group is adjusted so that its mean is 0 and its
    variance is 1, while preserving the relative gaps between the values. It
    then scales the values by the weight and adds the bias.

    The result is a tensor of the same shape as the input."""

    groups = 1 if groups is None else groups
    layer = torch.nn.GroupNorm(
        num_groups=groups, num_channels=input.shape[-1], device="cpu"
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)


def convolution(
    weight: torch.Tensor, bias: torch.Tensor, input: torch.Tensor
) -> torch.Tensor:
    """Applies a 2D convolution to the incoming data, which should be a 4D
    tensor of shape [batch, channels, height, width]. The weight must be a 4D
    tensor of shape [output_channels, input_channels, kernel_height,
    kernel_width]. The bias must be a 1D tensor of shape [output_channels]. The
    result is a 4D tensor of shape [batch, output_channels, height, width].

    The kernel is run over every position in the height/width matrix, and
    outputs a single value for that position using the weights and biases."""

    layer = torch.nn.Conv2d(
        in_channels=weight.shape[1],
        out_channels=weight.shape[0],
        kernel_size=weight.shape[2:],
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer(input)
