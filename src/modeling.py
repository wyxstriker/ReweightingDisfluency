import torch.nn as nn
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraModel
from torch.nn import CrossEntropyLoss
import torch


class ElectraForSequenceDisfluency_real(ElectraPreTrainedModel):

    base_model_prefix = "discriminator"

    def __init__(self, config, num_labels, num_labels_tagging):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_labels_tagging = num_labels_tagging
        self.discriminator = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_pair = nn.Linear(config.hidden_size, num_labels)
        self.classifier_tagging = nn.Linear(config.hidden_size, num_labels_tagging)  # pretrain on auto data
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(self.device)
        self.alpha.data.fill_(0.5)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_tagging=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import AlbertTokenizer, AlbertForTokenClassification
        import torch
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertForTokenClassification.from_pretrained('albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """
        discriminator_hidden_states = self.discriminator(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        sequence_output = discriminator_hidden_states[0]
        pooled_output = discriminator_hidden_states[0][:,0]
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)
        logits_pair = self.classifier_pair(pooled_output)
        logits_tagging = self.classifier_tagging(sequence_output)

        if labels_tagging is not None or labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            total_loss = None
            if labels_tagging is not None:
                loss_tagging = loss_fct(logits_tagging.view(-1, self.num_labels_tagging), labels_tagging.view(-1))
                total_loss = loss_tagging
            if labels is not None:
                loss_pair = loss_fct(logits_pair.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss_pair
                else:
                    total_loss += loss_pair
            return total_loss
        else:
            return logits_pair, logits_tagging



class ElectraForSequenceDisfluency_sing(ElectraPreTrainedModel):

    base_model_prefix = "discriminator"

    def __init__(self, config, num_labels, num_labels_tagging):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_labels_tagging = num_labels_tagging
        self.num_labels_sing = 2
        self.discriminator = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_pair = nn.Linear(config.hidden_size, num_labels)
        self.classifier_tagging = nn.Linear(config.hidden_size, num_labels_tagging)  # pretrain on auto data
        self.classifier_sing = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_tagging=None,
        labels_sing=None,
        labels_lm=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import AlbertTokenizer, AlbertForTokenClassification
        import torch
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertForTokenClassification.from_pretrained('albert-base-v2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """
        discriminator_hidden_states = self.discriminator(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds
        )
        sequence_output = discriminator_hidden_states[0]
        pooled_output = discriminator_hidden_states[0][:,0]
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)
        logits_pair = self.classifier_pair(pooled_output)
        logits_tagging = self.classifier_tagging(sequence_output)
        logits_sing = self.classifier_sing(pooled_output)

        if labels_tagging is not None or labels is not None or labels_sing is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            total_loss = None
            if labels_tagging is not None:
                loss_tagging = loss_fct(logits_tagging.view(-1, self.num_labels_tagging), labels_tagging.view(-1))
                total_loss = loss_tagging
            if labels is not None:
                loss_pair = loss_fct(logits_pair.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss_pair
                else:
                    total_loss += loss_pair
            if labels_sing is not None:
                loss_sing = loss_fct(logits_sing.view(-1, self.num_labels_sing), labels_sing.view(-1))
                if total_loss is None:
                    total_loss = loss_sing
                else:
                    total_loss += loss_sing
            return total_loss
        else:
            return logits_pair, logits_tagging, logits_sing