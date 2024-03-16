from transformers import XLNetModel, AutoTokenizer
import torch


class XLNetClassifier(torch.nn.Module):
    """
    XLNet-based classifier model for binary classification.

    Args:
        checkpoint (str): The name or path of the pre-trained XLNet checkpoint.
        dropout (float, optional): Dropout probability. If None, defaults to the value specified in the XLNet configuration. Default is None.

    Attributes:
        checkpoint (str): The name or path of the pre-trained XLNet checkpoint.
        tokenizer (AutoTokenizer): XLNet tokenizer.
        xlnet (XLNetModel): Pre-trained XLNet model.
    """

    def __init__(self, checkpoint, dropout=None):
        """
        Initializes the XLNetClassifier model.

        Args:
            checkpoint (str): The name or path of the pre-trained XLNet checkpoint.
            dropout (float, optional): Dropout probability. If None, defaults to the value specified in the XLNet configuration. Default is None.
        """
        super(XLNetClassifier, self).__init__()
        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.xlnet = XLNetModel.from_pretrained(checkpoint, num_labels=2)
        self.linear1 = torch.nn.Linear(
            self.xlnet.config.hidden_size, self.xlnet.config.hidden_size)
        self.tanh = torch.nn.Tanh()

        self.dropout = torch.nn.Dropout(
            dropout if dropout else self.xlnet.config.summary_last_dropout)
        self.linear2 = torch.nn.Linear(self.xlnet.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the XLNetClassifier model.

        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.

        Returns:
            torch.Tensor: Output logits.
        """
        xlnet_output = self.xlnet(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        summary = self.linear1(xlnet_output.last_hidden_state[:, -1])
        activation = self.tanh(summary)
        dropout_output = self.dropout(activation)
        output = self.linear2(dropout_output)

        return output
