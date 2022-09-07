import torch.nn as nn
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.electra.modeling_electra import ElectraModel
from torch.nn import CrossEntropyLoss


class ElectraForSequenceDisfluency_real(ElectraPreTrainedModel):

    base_model_prefix = "discriminator"

    def __init__(self, config, num_labels, num_labels_tagging):
        super().__init__(config)
        self.num_labels = num_labels
        self.num_labels_tagging = num_labels_tagging
        self.discriminator = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_pair = nn.Linear(config.hidden_size, num_labels)
        self.classifier_tagging = nn.Linear(config.hidden_size, num_labels_tagging)
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