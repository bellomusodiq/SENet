import torch
import torch.nn as nn

from .models import DenseNet

CUDA = torch.cuda.is_available()

CUDA = torch.cuda.is_available()
conv_model = ConvNet(3, 10)
se_model = SENet(3, 8, 10)

if CUDA:
    conv_model = conv_model.cuda()
    se_modal = se_model.cuda()

# hyper parameters
learning_rate = 0.01
decay_rate = 0.5
batch_size = 150
conv_optimizer = torch.optim.SGD(
    conv_model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9
)
conv_loss_fn = nn.CrossEntropyLoss()

se_optimizer = torch.optim.SGD(
    se_model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9
)
se_loss_fn = nn.CrossEntropyLoss()

conv_train_losses = []
conv_train_accuracy = []
conv_test_losses = []
conv_test_accuracy = []

se_train_losses = []
se_train_accuracy = []
se_test_losses = []
se_test_accuracy = []

num_epochs = 250

if __name__ == __main__:
    # training loop for conv model
    for epoch in range(num_epochs):
        epoch_train(
            conv_model,
            num_epochs,
            epoch,
            train_loader,
            conv_loss_fn,
            conv_optimizer,
            conv_train_losses,
            conv_train_accuracy,
        )
        epoch_test(
            conv_model,
            num_epochs,
            epoch,
            test_loader,
            conv_loss_fn,
            conv_test_losses,
            conv_test_accuracy,
        )

    # training loop for SE model
    for epoch in range(num_epochs):
        epoch_train(
            se_model,
            num_epochs,
            epoch,
            train_loader,
            se_loss_fn,
            se_optimizer,
            se_train_losses,
            se_train_accuracy,
        )
        epoch_test(
            se_model,
            num_epochs,
            epoch,
            test_loader,
            se_loss_fn,
            se_test_losses,
            se_test_accuracy,
        )
