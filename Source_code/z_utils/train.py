import os
import torch
import random
import time
import datetime
import numpy as np

from .evaluate import eval_model
from .plot import plot
from .global_constants import PATH_SAVED_MODELS, RANDOM_SEED, EVAL_FREQUENCY


def train_model(model, train_dataloader, val_dataloader, batch_size, loss_fn, optimizer, device, scheduler, epochs):
    """
    Trains the specified model on the training data and evaluates it on the validation data for a certain number of epochs.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        batch_size (int): Batch size for training.
        loss_fn (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device (CPU or GPU) to run the model and tensors.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Number of epochs for training.

    Returns:
        tuple: A tuple containing the following elements:
            - metrics (list): List of dictionaries containing training and validation metrics.
            - losses (list): List of dictionaries containing training losses.
            - sliding_accuracies (list): List of dictionaries containing sliding accuracies.
            - info_best_model (str): Information about the best performing model.
            - run_times (list): List of strings representing the run times for each epoch.
    """
    metrics = []
    losses = []
    sliding_accuracies = []
    run_times = []
    best_acc = 0
    info_best_model = ""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    total_time = time.monotonic_ns()

    for epoch_num in range(epochs):
        print(f"Epoch {epoch_num + 1} of {epochs} started.")
        print("Training ...")

        time0 = time.monotonic_ns()

        total_loss = 0
        correct_predictions = 0.0

        for i, data in enumerate(train_dataloader):
            model.train()

            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(
                device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(
                device, dtype=torch.long)
            labels = data['labels'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            num_data = (i + 1) * batch_size
            sliding_accuracies.append(
                {"accuracy": correct_predictions / num_data, "epoch": epoch_num})

            loss = loss_fn(outputs, labels)
            losses.append({"loss": loss, "epoch": epoch_num})
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if (i % EVAL_FREQUENCY == 0 and i > 0) or i == len(train_dataloader) - 1:
                train_acc = correct_predictions / num_data
                train_loss = total_loss / num_data
                val_acc, val_loss = eval_model(
                    model, val_dataloader, batch_size, loss_fn, device)
                metrics.append({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc,
                               "val_loss": val_loss, "epoch": epoch_num, "samples": num_data})

                if val_acc > best_acc:
                    if not os.path.exists(PATH_SAVED_MODELS):
                        os.makedirs(PATH_SAVED_MODELS)
                    torch.save(model.state_dict(
                    ), f"{PATH_SAVED_MODELS + model.checkpoint[model.checkpoint.find('/')+1:]}.bin")
                    info_best_model = f"Epoch: {epoch_num}, Sample: {num_data}"
                    best_acc = val_acc

        elapsed_time = datetime.timedelta(
            microseconds=(time.monotonic_ns() - time0)/1000)
        run_times.append(str(elapsed_time))

    elapsed_time = datetime.timedelta(
        microseconds=(time.monotonic_ns() - total_time)/1000)
    run_times.append(str(elapsed_time))

    return metrics, losses, sliding_accuracies, info_best_model, run_times
