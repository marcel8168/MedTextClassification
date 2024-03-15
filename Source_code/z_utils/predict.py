import torch
from tqdm import tqdm


def predict(model, texts, tokenizer, device, max_len=512):
    progress_bar = tqdm(range(len(texts)))

    predictions = []
    for data in texts:
        text = str(data)

        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
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

        probabilities = torch.sigmoid(logits.squeeze())
        predictions.append(probabilities)

        progress_bar.update(1)

    return predictions
