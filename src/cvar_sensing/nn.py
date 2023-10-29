import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layer: int,
        activation=torch.nn.LeakyReLU(),
        output_func=torch.nn.Identity(),
        bias: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.activation = activation
        self.output_func = output_func
        self.bias = bias

        modules = []
        if self.num_layer == 0:
            modules = []
        elif self.num_layer == 1:
            modules = [torch.nn.Linear(self.input_size, self.output_size, bias=self.bias)]
        elif self.num_layer >= 2:
            modules = [torch.nn.Linear(self.input_size, self.hidden_size, bias=self.bias), self.activation]
            for _ in range(0, self.num_layer - 2, 1):
                modules = modules + [torch.nn.Linear(self.hidden_size, self.hidden_size, bias=bias), self.activation]

            modules = modules + [torch.nn.Linear(self.hidden_size, self.output_size, bias=bias)]
        else:
            pass

        modules = modules + [self.output_func]

        self.nn = torch.nn.Sequential(*modules)

    def forward(self, x):
        x = self.nn(x)
        return x


class GRUCell(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.h0 = torch.nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_size)))

        self.rnn = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layer, batch_first=True)

    def forward(self, x, h0=None):
        if h0 is None:
            batch_size, _, _ = x.shape
            h0 = self.h0.expand(self.num_layer, batch_size, self.hidden_size).contiguous()
        # x is of the shape N x L x input_size
        out, h = self.rnn(x, h0)
        return out, h


class GRU(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.h0 = torch.nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_size)))

        self.rnn = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layer, batch_first=True)

    def forward(self, x):
        batch_size, _, _ = x.shape
        h0 = self.h0.expand(self.num_layer, batch_size, self.hidden_size).contiguous()
        # x is of the shape N x L x input_size
        out, _ = self.rnn(x, h0)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.h0 = torch.nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_size)))
        self.c0 = torch.nn.Parameter(torch.zeros((self.num_layer, 1, self.hidden_size)))

        self.rnn = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layer, batch_first=True)

    def forward(self, x):
        batch_size, _, _ = x.shape
        h0 = self.h0.expand(self.num_layer, batch_size, self.hidden_size).contiguous()
        c0 = self.c0.expand(self.num_layer, batch_size, self.hidden_size).contiguous()
        # x is of the shape N x L x input_size
        out, _ = self.rnn(x, (h0, c0))
        return out


class TimeSeriesEncoder(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layer: int):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        self.rnn = GRU(self.input_size, self.hidden_size, 1)
        self.mlp = MLP(self.hidden_size, self.output_size, self.hidden_size, self.num_layer)

    def forward(self, x):
        x = self.rnn(x)
        x = self.mlp(x)
        return x
