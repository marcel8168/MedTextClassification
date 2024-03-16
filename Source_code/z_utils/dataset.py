import torch


class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling text data.

    Args:
        texts (list): List of input texts.
        labels (list): List of corresponding labels.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text inputs.
        device (str): Device to place tensors on ('cpu' or 'cuda').
        max_len (int): Maximum length of input sequences after tokenization (default: 512).
    """

    def __init__(self, texts, labels, tokenizer, device, max_len=512):
        """
        Initialize the dataset.

        Args:
            texts (list): List of input texts.
            labels (list): List of corresponding labels.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text inputs.
            device (str): Device to place tensors on ('cpu' or 'cuda').
            max_len (int): Maximum length of input sequences after tokenization (default: 512).
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the encoded text input and corresponding label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = str(self.texts[idx])
        labels = self.labels[idx]

        encodings = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        ids = encodings['input_ids'].flatten()
        mask = encodings['attention_mask'].flatten()
        token_type_ids = encodings["token_type_ids"].flatten()

        return {
            'text': text,
            'input_ids': ids.to(self.device),
            'attention_mask': mask.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'labels': torch.tensor(labels, dtype=torch.long).to(self.device)
        }
