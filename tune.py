import numpy as np
import torch
from ax.service.managed_loop import optimize

from model import AutoEncoder
from main import load_dataset

def train(dataloader, parameters, device):
    model = AutoEncoder(input_dim=1900, nlayers=parameters.get('nlayers', 5), latent=100)
    model = model.to(device)

    model.train()
    train_loss = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.get('lr', 1e-5), 
            weight_decay=parameters.get('weight_decay', 0.))
    loss_func = torch.nn.MSELoss()

    for epoch in range(parameters.get('epochs', 1000)):
        for index, (data, ) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, data)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

    return model

def test(dataloader, model):
    model.eval()
    test_loss = 0

    loss_func = torch.nn.MSELoss()

    for index, (data, ) in enumerate(dataloader, 1):
        with torch.no_grad():
            output = model(data)
        loss = loss_func(output, data)
        test_loss += loss.item()

    return test_loss / index

def train_test(parameterization):
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    train_dataloader, test_dataloader = load_dataset('data/aponc_sda.npz', batch_size, device)

    net = train(train_dataloader, parameterization, device)

    return test(test_dataloader, net)

def tune():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {'name': 'lr', 'type': 'range', 'bounds': [1e-6, 0.4], 'log_scale': True},
            {'name': 'weight_decay', 'type': 'range', 'bounds': [0.0, 1.0], 'log_scale': False},
            {'name': 'nlayers', 'type': 'range', 'bounds': [2, 6], 'log_scale': False},
            #{'name': 'momentum', 'type': 'range', 'bounds': [0.0, 1.0]},
        ],
        evaluation_function=train_test,
        objective_name='mse_loss',
    )

    print(best_parameters)
    print('means, covariances', values)

    return experiment

def best(experiment):
    df = experiment.fetch_data().df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]

    print(best_arm)

    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100
    train_dataloader, test_dataloader = load_dataset('data/aponc_sda.npz', batch_size, device)

    combined_train_test_set = torch.utils.data.ConcatDataset([
        train_dataloader.dataset, 
        test_dataloader.dataset,
    ])

    combined_train_test_loader = torch.utils.data.DataLoader(
        combined_train_test_set, 
        batch_size=batch_size, 
        shuffle=True,
    )

    net = train(train_dataloader, best_arm.parameters, device)

    test_mse_loss = test(test_dataloader, net)

    print('MSE loss (test set): %f' % (test_mse_loss))

def main():
    torch.manual_seed(123)

    experiment = tune()
    best(experiment)

if __name__ == '__main__':
    main()
