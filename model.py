from transformers import BertModel,BertTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
from torchcrf import CRF
class BERT_NER(nn.Module):
	def __init__(self,args,label_map):
		super().__init__()
		self.args = args
		self.label_map = label_map
		self.bertmodel = BertModel.from_pretrained(args.pretrain_path)
		self.dropout = nn.Dropout(args.dropout)
		num_labels = len(self.label_map)
		self.classifier = nn.Linear(args.hidden_size, num_labels)

		self.crf = CRF(num_labels, batch_first=True)
	def forward(self,input_ids,attention_mask,label_ids,is_training = True):
		outputs = self.bertmodel(input_ids,attention_mask=attention_mask)
		sequence_output = outputs[0] # batch sen_len embedding_dim
		sequence_output = self.dropout(sequence_output)
		logits = self.classifier(sequence_output) # batch sen_len label_num

		if is_training:
			main_loss = - self.crf(logits, label_ids, mask=attention_mask.byte(), reduction='mean')
			return main_loss
		else:
			pred_logits = self.crf.decode(logits, mask=attention_mask.byte())
			return pred_logits
		
