import os
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import random
import numpy as np

from .model_evaluation import eval_model
from .global_constants import PATH_SAVED_MODELS, RANDOM_SEED


def train_model(model, train_dataloader, val_dataloader, batch_size, loss_fn, optimizer, device, scheduler, epochs):
    progress_bar = tqdm(range(len(train_dataloader) * epochs))
    history = []
    best_acc = 0

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    for epoch_num in range(epochs):
        print("_" * 30)
        print(f'Epoch {epoch_num} started.')

        losses = []
        validation_data = [[], []]
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

            loss = loss_fn(outputs, labels)
            losses.append(loss)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)

            if i % 50 == 0 and i > 0:
                val_acc, val_loss = eval_model(
                    model, val_dataloader, batch_size, loss_fn, device)
                validation_data[0].append(val_acc)
                validation_data[1].append(val_loss)
                if val_acc > best_acc:
                    if not os.path.exists(PATH_SAVED_MODELS):
                        os.makedirs(PATH_SAVED_MODELS)
                    torch.save(model.state_dict(
                    ), f"{PATH_SAVED_MODELS + model.checkpoint[model.checkpoint.find('/')+1:]}.bin")
                    best_acc = val_acc

        plot([torch.Tensor(losses).to('cpu')], ["training loss"],
             f"Training loss of epoch {epoch_num}", "Loss", "No. of training samples", [0, 0.1])

        num_data = len(train_dataloader) * batch_size
        train_acc = correct_predictions / num_data
        train_loss = total_loss / num_data
        print(
            f'Epoch: {epoch_num}, Train Accuracy {train_acc}, Loss:  {train_loss}')

        val_acc, val_loss = eval_model(
            model, val_dataloader, batch_size, loss_fn, device)
        validation_data[0].append(val_acc)
        validation_data[1].append(val_loss)
        print(
            f'Epoch: {epoch_num}, Validation Accuracy {val_acc}, Loss:  {val_loss}')
        plot([torch.Tensor(validation_data[0])], ["validation accuracy"],
             f"Validation accuracy of epoch {epoch_num}", "Acc", "Batches of 50 training samples", [0.95, 1])

        history.append({"train_acc": train_acc, "train_loss": train_loss,
                       "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_acc:
            if not os.path.exists(PATH_SAVED_MODELS):
                os.makedirs(PATH_SAVED_MODELS)
            torch.save(model.state_dict(
            ), f"{PATH_SAVED_MODELS + model.checkpoint[model.checkpoint.find('/')+1:]}.bin")
            best_acc = val_acc

    return history


def plot(data_list: list, data_label_list: list[str], title, ylabel, xlabel, ylim=None):
    for i, data in enumerate(data_list):
        plt.plot(data, label=data_label_list[i])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim:
        plt.ylim(ylim)
    if len(data_list) > 1:
        plt.legend()
    plt.show()
    plt.close()
