from transformers import DebertaForSequenceClassification, AutoTokenizer
import torch


class DeBERTaClassifier(torch.nn.Module):
    """
    DeBERTa-based classifier model for binary classification.

    Args:
        checkpoint (str): The name or path of the pre-trained DeBERTa checkpoint.
        dropout (float): Dropout probability. Default is 0.1.

    Attributes:
        checkpoint (str): The name or path of the pre-trained DeBERTa checkpoint.
        tokenizer (AutoTokenizer): DeBERTa tokenizer.
        deberta (DebertaForSequenceClassification): Pre-trained DeBERTa model.
    """

    def __init__(self, checkpoint, dropout=0.1):
        """
        Initializes the DeBERTaClassifier model.

        Args:
            checkpoint (str): The name or path of the pre-trained DeBERTa checkpoint.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(DeBERTaClassifier, self).__init__()
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.deberta = DebertaForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the DeBERTaClassifier model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.

        Returns:
            torch.Tensor: Output logits.
        """
        output = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return output.logits
