from transformers import AdamW
from utils import liner_warmup,set_lr,clip_gradient
from torch.nn import CrossEntropyLoss
from seqeval.metrics import classification_report,f1_score
import fitlog

class Trainer:
	def __init__(self,args,logger,model,label_map,train_dataloader,dev_dataloader,test_dataloader):
		self.args = args
		self.logger = logger
		self.model = model
		self.label_map = label_map
		self.train_dataloader = train_dataloader
		self.dev_dataloader = dev_dataloader
		self.test_dataloader = test_dataloader
		self.best_dev = 0
		self.best_model = model
	def train(self):
		args = self.args
		self.model.train()
		optimizer = AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
		step = 0
		for epoch in range(args.epochs):
			total_step = len(self.train_dataloader)
			for i,batch in enumerate(self.train_dataloader):
				step += 1
				input_ids,attention_mask,label_ids = batch
				input_ids,attention_mask,label_ids = input_ids.to(args.device),\
													 attention_mask.to(args.device),label_ids.to(args.device)
				loss = self.model.forward(input_ids,attention_mask,label_ids,is_training=True)
				fitlog.add_loss(loss, name="Loss",step = step)
				train_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
	                epoch + 1, args.epochs, i + 1, total_step, loss.item())
				print(train_info)
				self.logger.info(train_info)

				# 对梯度的一些操作
				cur_step = i + 1 + epoch * total_step
				t_step = args.epochs * total_step
				liner_warm_rate = liner_warmup(cur_step, t_step, args.warmup)
				set_lr(optimizer, liner_warm_rate * args.lr)
				optimizer.zero_grad()
				loss.backward()
				clip_gradient(optimizer, args.grad_clip)
				optimizer.step()

			if (epoch+1) % args.eval_every == 0:
				f1_dev = self.eval()
				dev_info = 'DEV  f1_dev:{}'.format(f1_dev)
				fitlog.add_metric({"dev": {"f1": f1_dev}},step = step)
				print(dev_info)
				self.logger.info(dev_info)
				if f1_dev > self.best_dev:
					self.best_dev = f1_dev
					self.best_model = self.model
					fitlog.add_best_metric({"test": {"f1": self.best_dev}})
		f1_test = self.test()
		fitlog.add_best_metric({"test": {"f1": f1_test}})
		test_info = 'TEST  f1_test:{}'.format(f1_test)
		print(test_info)
		self.logger.info(test_info)
		fitlog.finish()  # finish the logging

	def eval(self):
		args = self.args
		self.model.eval()
		toty_pred = []
		toty_true = []
		for i,batch in enumerate(self.dev_dataloader):
			input_ids, attention_mask, label_ids = batch
			input_ids, attention_mask, label_ids = input_ids.to(args.device), \
												   attention_mask.to(args.device), label_ids.to(args.device)
			pred_logits = self.model.forward(input_ids,attention_mask,label_ids,is_training=False)
			label_ids = label_ids.to('cpu').numpy()
			attention_mask = attention_mask.to('cpu').numpy()
			y_pred,y_true = self.transform_NER(pred_logits,label_ids,attention_mask)
			toty_pred.extend(y_pred)
			toty_true.extend(y_true)
		f1 = f1_score(toty_true, toty_pred)
		self.model.train()
		return f1

	def test(self):
		args = self.args
		self.best_model.eval()
		toty_pred = []
		toty_true = []
		for i,batch in enumerate(self.test_dataloader):
			input_ids, attention_mask, label_ids = batch
			input_ids, attention_mask, label_ids = input_ids.to(args.device), \
												   attention_mask.to(args.device), label_ids.to(args.device)
			pred_logits = self.best_model.forward(input_ids,attention_mask,label_ids,is_training=False)
			label_ids = label_ids.to('cpu').numpy()
			attention_mask = attention_mask.to('cpu').numpy()
			y_pred,y_true = self.transform_NER(pred_logits,label_ids,attention_mask)
			toty_pred.extend(y_pred)
			toty_true.extend(y_true)
		f1 = f1_score(toty_true, toty_pred)
		return f1

	def transform_NER(self,pred_logits,label_ids,attention_mask):
		reverse_label_map = {value:key for key,value in self.label_map.items()}
		y_pred = []
		y_true = []
		for i, mask in enumerate(attention_mask):
			temp_true = []
			temp_pred = []
			for j, m in enumerate(mask):
				if j == 0:
					continue
				if m:
					if reverse_label_map[label_ids[i][j]] != "X":
						temp_true.append(reverse_label_map[label_ids[i][j]])
						temp_pred.append(reverse_label_map[pred_logits[i][j]])
				else:
					break
			y_pred.append(temp_pred)
			y_true.append(temp_true)
		return y_pred,y_true