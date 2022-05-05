import numpy as np
from transformers import BertTokenizer,RobertaTokenizer
import torch
import pickle as pkl
import os
from torch.utils.data import Dataset

def load_data(file_path):
	total_data = []
	total_label = []
	with open(file_path,'r',encoding='utf-8') as f:
		temp_data = []
		temp_label = []
		line = f.readlines()
		for i,l in enumerate(line):
			if i <= 1:
				continue
			elif l == '\n':
				total_data.append(temp_data)
				total_label.append(temp_label)
				temp_label = []
				temp_data = []
				continue
			else:
				l = l.strip()
				temp_list = l.split(' ')
				word = temp_list[0]
				label = temp_list[-1]
				temp_data.append(word)
				if label == 'B-MISC':
					label = 'B-OTHER'
				elif label == 'I-MISC':
					label = 'I-OTHER'
				temp_label.append(label)

	return total_data,total_label

def load_data_twitter(file_path):
	total_data = []
	total_label = []
	with open(file_path, 'r', encoding='utf-8') as f:
		line = f.readlines()
		for i, l in enumerate(line):
			l = l.strip()
			if i % 3 == 0:
				temp_data = l.split(' ')
				total_data.append(temp_data)
			elif i%3 == 1:
				temp_label = l.split(' ')
				refined_label = []
				for label in temp_label:
					if label == 'B-MISC':
						new_label = 'B-OTHER'
					elif label == 'I-MISC':
						new_label = 'I-OTHER'
					else:
						new_label = label
					refined_label.append(new_label)
				total_label.append(refined_label)
	return total_data,total_label



class Prepare:
	def __init__(self,args,label_list,total_data,total_label):
		self.args = args
		self.total_data = total_data
		self.label_list = label_list
		self.total_label = total_label
		self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
		self.label_list = label_list
	'''
	def calculate_max_len(self):
		max_len = 0
		for i in range(len(self.total_label)):
			if len(self.total_label[i]) > max_len:
				max_len = len(self.total_label[i])
	
		return max_len
	'''
	def prepare_data(self):

		print('preparing...')
		max_seq_length = 0
		# max_len = self.calculate_max_len()
		label_map = {label:i for i,label in enumerate(self.label_list)}
		# tokenize 加入X 记录最大长度
		tokenss = [] #所有句子切分成子词之后的token，没有转换为id
		rlabels = []
		for i in range(len(self.total_data)):
			sentence = self.total_data[i]
			label = self.total_label[i]

			tokens = [] #切分成子词后的一句话
			rlabel = [] #[[@],[@,#],...]

			for j,word in enumerate(sentence):
				token = self.tokenizer.tokenize(word)
				tokens.extend(token)
				label_temp = label[j]

				for m in range(len(token)): #有没有子词
					if m == 0:
						rlabel.append(label_temp)
					else:
						rlabel.append("X")

			if len(tokens) > max_seq_length:
				max_seq_length = len(tokens)

			tokenss.append(tokens)
			rlabels.append(rlabel)

		#转换为id
		all_input_ids = []
		all_attention_mask = []
		all_label_ids = []

		for i in range(len(self.total_data)):
			ntokens = []
			label_ids = []
			# cls
			ntokens.append("[CLS]")
			label_ids.append(label_map["[CLS]"])
			# 加入所有token
			tokens = tokenss[i]
			rlabel = rlabels[i]
			for j, token in enumerate(tokens):
				ntokens.append(token)
				label_ids.append(label_map[rlabel[j]])
			# 转换为id
			input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
			input_mask = [1] * len(input_ids)
			# pad
			while len(input_ids) < max_seq_length + 1: #因为加了CLS
				input_ids.append(self.tokenizer.convert_tokens_to_ids('[PAD]'))
				input_mask.append(0)
				label_ids.append(label_map["[PAD]"])

			all_input_ids.append(input_ids)
			all_attention_mask.append(input_mask)
			all_label_ids.append(label_ids)

		input_ids = torch.tensor(all_input_ids,dtype=torch.long)
		attention_mask = torch.tensor(all_attention_mask,dtype=torch.long)
		label_ids = torch.tensor(all_label_ids,dtype=torch.long)
		data_dict = {'attention_mask':attention_mask,'input_ids':input_ids,
                    'label_ids':label_ids,'label_map':label_map}
		return data_dict

def transform_data(total_data,total_label):
	with open('data/test_t.txt','w',encoding='utf-8') as f:
		for i in range(len(total_data)):
			data = ' '.join(total_data[i])
			label = ' '.join(total_label[i])
			f.write(data)
			f.write('\n')
			f.write(label)
			f.write('\n')

class NER_Dataset(Dataset):
	def __init__(self,data_dict) -> None:
		super().__init__()
		self.data_dict = data_dict
	def __getitem__(self,index):
		input_ids = self.data_dict['input_ids'][index]
		attention_mask = self.data_dict['attention_mask'][index]
		label_ids = self.data_dict['label_ids'][index]
		return input_ids,attention_mask,label_ids
	def __len__(self):
		return len(self.data_dict['label_ids'])

if __name__ == '__main__':

	total_data,total_label = load_data('data/test.txt')
	print(total_data)
	print(total_label)
	# transform_data(total_data,total_label)