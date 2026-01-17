import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 01 -  Multi-Layer Perceptron

    ---

    Implement and compare different activation functions (Sigmoid, Tanh, ReLU) in a MLP to classify loan applications as approved or rejected based on applicant data.

    Take "Loan Prediction Dataset" from [kaggle](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling)

    ---

    Submitted by : Yash Pravin Pawar

    TY-A3 - 371079
    """)
    return


@app.cell
def _():
    import polars as pl

    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    import torch 
    from torch import nn, optim
    return StandardScaler, nn, optim, pl, plt, sns, torch, train_test_split


@app.cell
def _(pl):
    df = pl.read_excel("./Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data",read_options={"header_row": 0})
    df
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Descriptions

    | ID |	Customer ID |
    | :---- | :---- |
    |Age	|Customer's age in completed years|
    |Experience|	#years of professional experience|
    |Income	|Annual income of the customer ($000)|
    |ZIPCode|	Home Address ZIP code.|
    |Family	|Family size of the customer|
    |CCAvg	Avg. |spending on credit cards per month ($000)|
    |Education	|Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional|
    |Mortgage	|Value of house mortgage if any. ($000)|
    |Personal Loan	|Did this customer accept the personal loan offered in the last campaign?|
    |Securities |Account	Does the customer have a securities account with the bank?|
    |CD Account	|Does the customer have a certificate of deposit (CD) account with the bank?|
    |Online	|Does the customer use internet banking facilities?|
    |CreditCard|	Does the customer use a credit card issued by UniversalBank?|

    **Goal:** to find if the customer is accepted for personal loan, a binary classification problem
    """)
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Personal Loan', y='Income', data=df, palette='Set2')
    plt.title('Income Distribution by Loan Acceptance')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Income', y='CCAvg', hue='Personal Loan', data=df, alpha=0.6)
    plt.title('Income vs. Average Credit Card Spending')
    plt.show()
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Customer Age')
    plt.show()
    return


@app.cell
def _(df, plt):
    edu_counts = df.to_pandas()['Education'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(edu_counts.index.astype(str), edu_counts.values, color='salmon')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.title('Count of Customers by Education Level')
    plt.show()
    return


@app.cell
def _(df, pl):
    eda_summary = df.group_by("Personal Loan").agg([
        pl.col("Income").mean().alias("Avg_Income"),
        pl.col("CCAvg").mean().alias("Avg_CC_Spending"),
        pl.col("CD Account").mean().alias("CD_Account_Rate")
    ])

    eda_summary
    return


@app.cell
def _(df, pl):
    # df.describe() shows that experience has min of -3, which is impossible, so use polars, to make it 0
    df_clean = df.with_columns(
        pl.when(pl.col("Experience") < 0).then(0).otherwise(pl.col("Experience")).alias("Experience")
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Build the Dataset for the MLP
    """)
    return


@app.cell
def _(StandardScaler, df, torch, train_test_split):
    X = df.drop(["ID", "ZIP Code", "Personal Loan"]).to_numpy()
    y = df["Personal Loan"].to_numpy()

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_ts = torch.FloatTensor(X_train)
    y_train_ts = torch.FloatTensor(y_train).view(-1, 1)
    X_test_ts = torch.FloatTensor(X_test)
    y_test_ts = torch.FloatTensor(y_test).view(-1, 1)
    return X_test_ts, X_train_ts, y_test_ts, y_train_ts


@app.cell
def _(nn):
    # Build the MLP
    class LoanMLP(nn.Module):
        def __init__(self, input_size, activation_type='relu'):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 16),
                self._get_activation(activation_type),
                nn.Linear(16, 8),
                self._get_activation(activation_type),
                nn.Linear(8, 1),
                nn.Sigmoid() # Always Sigmoid at the end for binary classification
            )
        
        def _get_activation(self, name):
            if name == 'relu': return nn.ReLU()
            if name == 'tanh': return nn.Tanh()
            if name == 'sigmoid': return nn.Sigmoid()
            return nn.ReLU()

        def forward(self, x):
            return self.net(x)
    return (LoanMLP,)


@app.cell
def _(LoanMLP, X_test_ts, X_train_ts, nn, optim, torch, y_test_ts, y_train_ts):
    results_loss = {}
    results_acc = {}

    activations = ['relu', 'tanh', 'sigmoid']

    for act in activations:
        # Initialize model, loss, and optimizer
        model = LoanMLP(input_size=X_train_ts.shape[1], activation_type=act)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    
        losses = []
        epochs = 100
    
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train_ts)
            loss = criterion(outputs, y_train_ts)
        
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
        # Save loss history
        results_loss[act] = losses
    
        # Calculate and save final accuracy
        with torch.no_grad():
            y_pred = model(X_test_ts)
            y_pred_cls = y_pred.round()
            accuracy = (y_pred_cls.eq(y_test_ts).sum().item() / y_test_ts.shape[0])
            results_acc[act] = accuracy

    print("Training complete. Accuracies:", results_acc)
    return activations, results_acc, results_loss


@app.cell
def _(activations, plt, results_acc, results_loss):
    plt.figure(figsize=(14, 5))

    # Plot 1: Training Loss (Convergence Speed)
    plt.subplot(1, 2, 1)
    for act_ in activations:
        plt.plot(results_loss[act_], label=f'Activation: {act_}')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Final Accuracy (Performance)
    plt.subplot(1, 2, 2)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    plt.bar(results_acc.keys(), results_acc.values(), color=colors)
    plt.ylim(0.85, 1.0) # Zoom in to see the differences
    plt.title('Final Test Accuracy')
    plt.ylabel('Accuracy')

    # Annotate bars with the actual percentage
    for i, (act_, acc) in enumerate(results_acc.items()):
        plt.text(i, acc + 0.002, f"{acc:.2%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
