import torch
from tqdm import tqdm
import time
import datetime
import torch.nn.functional as F


def predict(model, texts, tokenizer, device="cpu", max_len=512):
    time0 = time.monotonic_ns()

    predictions = []
    label_list = []

    for data in texts:
        text = str(data)

        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True
        )
        ids = torch.tensor(inputs['input_ids'],
                           dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(inputs['attention_mask'],
                            dtype=torch.long).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(
            inputs["token_type_ids"], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(ids, mask, token_type_ids)

        if device == "cpu":
            probabilities = F.softmax(
                logits.squeeze(), -1).cpu().detach().numpy()
        else:
            probabilities = F.softmax(logits.squeeze(), -1)

        predictions.append(probabilities)

        if device == "cpu":
            label_list.append(data["labels"].cpu().detach().numpy())
        else:
            label_list.append(data["labels"])

        elapsed_time = datetime.timedelta(
            microseconds=(time.monotonic_ns() - time0)/1000)

    return predictions, label_list, elapsed_time
