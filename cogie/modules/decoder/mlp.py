from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)
        self.reset_parameters()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class SharedDropout(nn.Module):
    """
        SharedDropout differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
        Args:
            p (float):
                The probability of an element to be zeroed. Default: 0.5.
            batch_first (bool):
                If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
                Default: ``True``.
        Examples:
            >>> x = torch.ones(1, 3, 5)
            >>> nn.Dropout()(x)
            tensor([[[0., 2., 2., 0., 0.],
                     [2., 2., 0., 2., 2.],
                     [2., 2., 2., 2., 0.]]])
            >>> SharedDropout()(x)
            tensor([[[2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.],
                     [2., 0., 2., 0., 2.]]])
        """

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training:
            return x
        if self.batch_first:
            mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
        else:
            mask = self.get_mask(x[0], self.p)
        x = x * mask
        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)
