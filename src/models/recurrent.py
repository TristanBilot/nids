import torch
from torch import nn


class GRU(nn.Module):
    """
    GRU Class; very simple and lightweight
    """

    def __init__(self, x_dim, h_dim, z_dim, device, hidden_units=2):
        """
        x_dim : int
            The input dimension
        h_dim : int
            The hidden dimension
        z_dim : int
            The output dimension
        hidden_units : int
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        """
        super(GRU, self).__init__()

        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.hidden_units = hidden_units

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.device = device

        self.reset_state()

    def reset_state(self):
        self.hidden = self._init_hidden()

    def detach_state(self):
        self.hidden = torch.autograd.Variable(self.hidden)

    def _init_hidden(self):
        return torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(
            self.device
        )

    def forward(self, xs, h0=None, include_h=False):
        """
        Forward method for GRU

        xs : torch.Tensor
            The T x N x X_dim input of node embeddings
        h0 : torch.Tensor
            A hidden state for the GRU
        include_h : bool
            If true, return hidden state as well as output
        """
        # xs = self.drop(xs)

        xs, self.hidden = self.rnn(xs, self.hidden)

        if not include_h:
            return self.lin(xs)

        return self.lin(xs), self.hidden


class LSTM(GRU):
    """
    Slightly more complex RNN, but about equal at most tasks, though
    some papers show that LSTM is better in some instances than GRU

    Best practice to use LSTM first, and if GRU performs as well to switch to that
    """

    def __init__(self, x_dim, h_dim, z_dim, device, hidden_units=1):
        """
        Constructor for LSTM model

        x_dim : int
            The input dimension
        h_dim : int
            The hidden dimension
        z_dim : int
            The output dimension
        hidden_units : int
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        """
        super(LSTM, self).__init__(
            x_dim, h_dim, z_dim, device, hidden_units=hidden_units
        )

        # Just swapping out one component with another
        self.rnn = nn.LSTM(x_dim, h_dim, num_layers=hidden_units)

    def detach_state(self):
        self.hidden = (
            torch.autograd.Variable(self.hidden[0]),
            torch.autograd.Variable(self.hidden[1]),
        )

    def _init_hidden(self):
        return (
            torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(
                self.device
            ),
            torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(
                self.device
            ),
        )
