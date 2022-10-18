import torch

CUDA = torch.cuda.is_available()


def epoch_train(
    model,
    num_epochs,
    epoch,
    data_loader,
    loss_fn,
    optimizer,
    train_losses,
    train_accuracy,
):
    CUDA = torch.cuda.is_available()
    model.train()
    sum_loss = 0
    sum_accuracy = 0
    iteration = 0
    total_count = 0
    for i, (images, targets) in enumerate(data_loader):
        if CUDA:
            images = images.cuda()
            targets = targets.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        if CUDA:
            sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
        else:
            sum_accuracy += (targets == predicted).sum()
        loss = loss_fn(outputs, targets)
        sum_loss += loss.item() * len(targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1
        total_count += len(images)
        if i % 10 == 0:
            print(
                "Epoch {}/{}, Iteration: {}/{}, Training loss: {:.3f}, Accuracy: {:.3f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(data_loader),
                    sum_loss / iteration,
                    sum_accuracy / total_count,
                )
            )
    train_losses.append(sum_loss / iteration)
    train_accuracy.append(sum_accuracy / total_count)


def epoch_test(
    model, num_epochs, epoch, data_loader, loss_fn, test_losses, test_accuracy
):
    CUDA = torch.cuda.is_available()
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        sum_accuracy = 0
        iteration = 0
        total_count = 0
        for i, (images, targets) in enumerate(data_loader):
            if CUDA:
                images = images.cuda()
                targets = targets.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            if CUDA:
                sum_accuracy += (targets.cpu() == predicted.cpu()).sum()
            else:
                sum_accuracy += (targets == predicted).sum()
            loss = loss_fn(outputs, targets)
            sum_loss += loss.item() * len(targets)
            iteration += 1
            total_count += len(images)
            print(
                "Epoch {}/{}, Iteration: {}/{}, Test loss: {:.3f}, Accuracy: {:.3f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    len(data_loader),
                    sum_loss / iteration,
                    sum_accuracy / total_count,
                )
            )
        test_losses.append(sum_loss / iteration)
        test_accuracy.append(sum_accuracy / total_count)
