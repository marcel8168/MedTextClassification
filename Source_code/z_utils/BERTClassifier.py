from transformers import AutoModel, AutoTokenizer
import torch


class BERTClassifier(torch.nn.Module):
    """
    BERT-based classifier model for binary classification.

    Args:
        checkpoint (str): The name or path of the pre-trained BERT checkpoint.
        dropout (float): Dropout probability. Default is 0.1.

    Attributes:
        checkpoint (str): The name or path of the pre-trained BERT checkpoint.
        tokenizer (AutoTokenizer): BERT tokenizer.
        bert (AutoModel): Pre-trained BERT model.
        dropout (torch.nn.Dropout): Dropout layer.
        linear2 (torch.nn.Linear): Fully connected layer for classification.
    """

    def __init__(self, checkpoint, dropout=0.1):
        """
        Initializes the BERTClassifier model.

        Args:
            checkpoint (str): The name or path of the pre-trained BERT checkpoint.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(BERTClassifier, self).__init__()

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.bert = AutoModel.from_pretrained(checkpoint, num_labels=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Defines the forward pass of the BERTClassifier model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask. Default is None.
            token_type_ids (torch.Tensor): Token type ids. Default is None.

        Returns:
            torch.Tensor: Output logits.
        """
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        output = self.linear2(dropout_output)

        return output
