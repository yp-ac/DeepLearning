import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    from torch import nn
    from torch import optim
    return nn, optim, torch


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device
    return


@app.cell
def _(torch):
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    return x, y


@app.cell
def _(nn):
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    return (model,)


@app.cell
def _(model, nn, optim):
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return loss_function, optimizer


@app.cell
def _(loss_function, model, optimizer, x, y):
    # training loop
    for epoch in range(1000):
        prediction = model(x)
        loss = loss_function(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % 100) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return


@app.cell
def _(model, torch):
    test_input = torch.tensor([[5.0]])

    predicted_output = model(test_input)

    print("Prediction for x=5:", predicted_output.item())
    return


@app.cell
def _(torch):
    x_test = torch.tensor([[5.0], [7.0], [10.0], [-3.0]])
    return (x_test,)


@app.cell
def _(model, x_test):
    model(x_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
