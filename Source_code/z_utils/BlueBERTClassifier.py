from transformers import AutoModel, AutoTokenizer
import torch


class BlueBERTClassifier(torch.nn.Module):
    """
    BlueBERT-based classifier model for binary classification.

    Args:
        checkpoint (str): The name or path of the pre-trained BlueBERT checkpoint.
        dropout (float): Dropout probability. Default is 0.1.

    Attributes:
        checkpoint (str): The name or path of the pre-trained BlueBERT checkpoint.
        tokenizer (AutoTokenizer): BlueBERT tokenizer.
        bluebert (AutoModel): Pre-trained BlueBERT model.
        dropout (torch.nn.Dropout): Dropout layer.
        linear2 (torch.nn.Linear): Fully connected layer for classification.
    """

    def __init__(self, checkpoint, dropout=0.1):
        """
        Initializes the BlueBERTClassifier model.

        Args:
            checkpoint (str): The name or path of the pre-trained BlueBERT checkpoint.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(BlueBERTClassifier, self).__init__()

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        self.bluebert = AutoModel.from_pretrained(checkpoint, num_labels=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(self.bluebert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the BlueBERTClassifier model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.

        Returns:
            torch.Tensor: Output logits.
        """
        bluebert_output = self.bluebert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dropout_output = self.dropout(bluebert_output.pooler_output)
        output = self.linear2(dropout_output)

        return output
