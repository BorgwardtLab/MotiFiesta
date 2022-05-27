import torch
from torch.nn.functional import normalize


class DictionaryLearning(torch.nn.Module):

    def __init__(self, dim, n_terms):
        super(DictionaryLearning, self).__init__()
        self.dim = dim
        self.n_terms = n_terms
        d = normalize(torch.rand((n_terms, dim),
                      dtype=torch.float,
                      requires_grad=True)
                      )
        self.dictionary = torch.nn.Parameter(d)

        self.attributor = self.build_attributor()


    def build_attributor(self):
        layers = torch.nn.ModuleList()
        layers.append(torch.nn.Linear(self.dim, self.n_terms))
        layers.append(torch.nn.Softmax())
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.attributor(x)

    def entropy_score(self, attributions):
        """ Something close to uniform gets a score of 0, something
        with 0 entropy gets a score of 1
        """
        # get uniform baseline
        uni_probas = torch.ones_like(attributions) / attributions.shape[1]
        ent_uni = torch.distributions.Categorical(uni_probas).entropy()

        ent = torch.distributions.Categorical(attributions).entropy()

        return torch.ones_like(ent) - ent / ent_uni

    def loss(self, x, attributions):
        reconstruct = torch.matmul(attributions, self.dictionary)
        ortho = torch.matmul(self.dictionary, self.dictionary.t())
        ortho_loss = torch.nn.MSELoss()(torch.eye(self.n_terms), ortho)
        rec_loss = torch.nn.MSELoss()(x, reconstruct)
        return ortho_loss + rec_loss
    pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy.random import multivariate_normal as mn
    import torch

    data_1 = torch.tensor(mn([1, 0], [[.01, 0], [0, .01]], 1000), dtype=torch.float)
    data_2 = torch.tensor(mn([0, 1], [[.01, 0], [0, .01]], 1000), dtype=torch.float)

    data = torch.cat([data_1, data_2])

    epochs = 1000
    n_terms = 2
    dim = 2

    model = DictionaryLearning(dim, n_terms)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        attributions = model(data)
        optimizer.zero_grad()
        h = model.entropy_score(attributions)
        loss = model.loss(data, attributions)
        loss.backward()
        optimizer.step()
