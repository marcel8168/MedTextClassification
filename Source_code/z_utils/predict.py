import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F


def predict(model, dataloader=None, texts=None, device="cpu", max_len=512):
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

    data_list = dataloader if dataloader else texts
    for data in data_list:
        if texts:
            data = model.tokenizer.encode_plus(
                data,
                None,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=True
            )
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
        if dataloader and "labels" in data.keys():
            label_list.append(data["labels"].cpu().detach().numpy())

    return predictions, label_list


def predict_svm(svm, vectorizer, texts):
    """
    Predicts class probabilities for the given texts using SVM classifier.

    Args:
        svm (SVC): SVM classifier object.
        vectorizer (Vectorizer): Vectorizer object to transform text data.
        texts (array-like): List or array containing text data.

    Returns:
        array: Array of predicted class probabilities.
    """
    vectorized_texts = vectorizer.transform(texts)
    decision = svm.decision_function(vectorized_texts)
    reshaped_decision = np.array(decision).reshape(-1, 1)

    return reshaped_decision
