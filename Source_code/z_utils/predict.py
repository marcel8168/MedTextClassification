import torch
from tqdm import tqdm
import torch.nn.functional as F


def predict(model, dataloader, device="cpu", max_len=512):
    """
    Perform predictions using the specified model on the given dataloader.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        dataloader (torch.utils.data.DataLoader): Dataloader containing the input data.
        device (str, optional): Device to perform inference on (e.g., "cpu", "cuda"). Default is "cpu".
        max_len (int, optional): Maximum sequence length. Default is 512.

    Returns:
        tuple: A tuple containing the predicted probabilities and the corresponding true labels.
    """
    predictions = []
    label_list = []

    for data in dataloader:

        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(
            device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(
            device, dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

        probabilities = F.softmax(
            logits.squeeze(), -1).cpu().detach().numpy()

        predictions.append(probabilities)
        label_list.append(data["labels"].cpu().detach().numpy())

    return predictions, label_list
