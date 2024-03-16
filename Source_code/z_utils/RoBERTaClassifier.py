from transformers import AutoModel, AutoTokenizer
import torch


class RoBERTaClassifier(torch.nn.Module):
    """
    RoBERTa-based classifier model for binary classification.

    Args:
        checkpoint (str): The name or path of the pre-trained RoBERTa checkpoint.
        dropout (float): Dropout probability. Default is 0.1.

    Attributes:
        checkpoint (str): The name or path of the pre-trained RoBERTa checkpoint.
        tokenizer (AutoTokenizer): RoBERTa tokenizer.
        roberta (AutoModel): Pre-trained RoBERTa model.
    """

    def __init__(self, checkpoint, dropout=0.1):
        """
        Initializes the RoBERTaClassifier model.

        Args:
            checkpoint (str): The name or path of the pre-trained RoBERTa checkpoint.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(RoBERTaClassifier, self).__init__()

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.roberta = AutoModel.from_pretrained(checkpoint, num_labels=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the RoBERTaClassifier model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.

        Returns:
            torch.Tensor: Output logits.
        """
        roberta_output = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dropout_output = self.dropout(roberta_output.pooler_output)
        output = self.linear2(dropout_output)

        return output
