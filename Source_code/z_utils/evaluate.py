import torch


def eval_model(model, dataloader, batch_size, loss_fn, device):
    model = model.eval()

    loss = 0.0
    correct_predictions = 0.0

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(
                device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(
                device, dtype=torch.long)
            labels = data["labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss += loss_fn(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()

    num_data = len(dataloader) * batch_size
    return correct_predictions / num_data, loss / num_data
