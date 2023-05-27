import copy
import datetime
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, BertForSequenceClassification, RobertaForSequenceClassification

from Models.BERT.dataset_for_bert import Dataset_BERT, Collect_FN
from Models.BERT.eval_model import Eval_Model_For_BERT
from Models.Base.trainer_base import Trainer_Base
from compent.checkpoint import CheckPointer_Normal
from compent.comm import synchronize, get_rank
from compent.metric_logger import MetricLogger
from compent.utils import move_to_device, reduce_loss_dict, get_memory_used

device = torch.device('cuda')

# some modifications added in this file, mainly for avoiding the influence of the train label.


class Trainer_BERT(Trainer_Base):
    def __init__(self, lm, cfg, logger, distributed, sentences_all):
        self.lm = lm
        super(Trainer_BERT, self).__init__(cfg = cfg, logger = logger, distributed = distributed)
        self.checkpointer = CheckPointer_Normal(cfg = cfg, logger = logger, rank = get_rank())
        #if (sentences_all.has_test_data):
        #    eval_sentence = sentences_all.test_sentence
        #    eval_labels = sentences_all.test_label
        #    self.logger.visdom_text(text = 'eval on test', win_name = 'eval')
        #else:
        eval_sentence = sentences_all.unlabeled_sentence
        eval_labels = sentences_all.unlabeled_GT_label
        #self.logger.visdom_text(text = 'eval on all data', win_name = 'eval')
        dataloader_eval = self.__build_dataloader__(eval_sentence, eval_labels, for_train = False)
        self.evaler_on_all = Eval_Model_For_BERT(self.cfg, self.logger, distributed=True, rank=self.rank,
                                                 dataloader_eval=dataloader_eval, lm=lm)
        if (sentences_all.has_test_data):
            test_sentence = sentences_all.test_sentence
            test_labels = sentences_all.test_label
            dataloader_eval = self.__build_dataloader__(test_sentence, test_labels, for_train=False)
            self.test_dataloader = dataloader_eval


    def test_model(self, checkpoint_name, inference_path):
        if self.lm.startswith("bert"):
            self.model = BertForSequenceClassification.from_pretrained(self.lm,
                                                                       num_labels=self.cfg.model.number_classes,
                                                                       gradient_checkpointing=True)
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(self.lm,
                                                                          num_labels=self.cfg.model.number_classes)
        self.checkpointer.load_from_filename(self.model, filename=checkpoint_name)
        # test and store pred
        self.model = self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            preds = []
            for batch in self.test_dataloader:
                batch = move_to_device(batch)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                if self.lm.startswith("bert"):
                    token_type_ids = batch['token_type_ids']
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_score = outputs.logits
                pred = torch.argmax(pred_score, dim=1).tolist()
                preds.extend(pred)
        json.dump(preds, open(f"{inference_path}/eval_labels.json", "w"))

    def __build_dataloader__(self, sentences, labels, for_train):
        collect_fn = Collect_FN(self.lm, labels is not None)
        dataset = Dataset_BERT(sentences, labels)
        sampler = DistributedSampler(dataset, shuffle = for_train)
        dataloader = DataLoader(dataset, batch_size = self.cfg.classifier.batch_size, sampler = sampler,
                                collate_fn = collect_fn)
        return dataloader

    def get_loader_from_origin_sentence(self, sentences, labels):
        sentences, labels = self.upsample_balance(sentences, labels)
        sample_number_per_class = self.get_classes_count(labels)
        self.logger.info('sample_number_per_class:{}'.format(sample_number_per_class))
        dataloader_train = self.__build_dataloader__(sentences, labels, for_train = True)
        return dataloader_train

    def train_model(self, sentences, labels, ITR, finetune_from_pretrain = True):
        assert finetune_from_pretrain == True

        # train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels,
        #                                                                               test_size = 0.1)
        # dataloader_pseudo_eval = self.__build_dataloader__(test_sentences, test_labels, for_train = False)
        # self.eval_on_pseudo = Eval_Model_For_Long(self.cfg, self.logger, distributed = True, rank = self.rank,
        #                                           dataloader_eval = dataloader_pseudo_eval)

        dataloader_train = self.get_loader_from_origin_sentence(sentences, labels)
        sentences, labels = self.__do_train__(dataloader = dataloader_train, ITR = ITR)
        return sentences, labels

    def __do_train__(self, dataloader, ITR):
        self.logger.info('start training')

        self.logger.info('finetune from pretrain, load pretrain model')
        if self.lm.startswith("bert"):
            self.model = BertForSequenceClassification.from_pretrained(self.lm,  # 'bert-base-uncased',
                                                                       num_labels=self.cfg.model.number_classes,
                                                                       gradient_checkpointing=True,
                                                                       )
        else:
            self.model = RobertaForSequenceClassification.from_pretrained(self.lm,  # 'bert-base-uncased',
                                                                          num_labels=self.cfg.model.number_classes,
                                                                          #gradient_checkpointing=True,
                                                                          )
            #self.model._set_gradient_checkpointing(value=True)

        self.model.train()
        self.model = self.model.to(device)
        if (self.distributed):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
            )

        self.model.train()

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        stop_itr = self.get_stop_itr(ITR)

        optimizer = AdamW(self.model.parameters(), lr = self.cfg.classifier.lr)

        total_epoch = self.cfg.classifier.n_epochs
        total_itr = 0
        train_over_flag = False
        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            if (self.distributed):
                dataloader.sampler.set_epoch(epoch)
            for iteration, batch in enumerate(dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch = move_to_device(batch)
                optimizer.zero_grad()
                # input_ids: [16,128]   label_id:[16]
                del batch['sentences']
                output = self.model(**batch)
                loss = output.loss
                loss_dict_reduced = reduce_loss_dict({'loss_all': loss})
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss = losses_reduced, **loss_dict_reduced)
                loss.backward()
                optimizer.step()
                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)

                eta_seconds = meters.time.global_avg * (self.cfg.classifier.total_steps - iteration)
                eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))

                if total_itr % 100 == 0:
                    GPU_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    memory = get_memory_used()
                    self.logger.info(
                            meters.delimiter.join(
                                    [
                                        "eta: {eta}",
                                        "iter: {iter}",
                                        "total_itr: {total_itr}",
                                        "{meters}",
                                        "lr: {lr:.6f}",
                                        "max mem: {GPU_memory:.0f}",
                                        "memory:{memory:.1f}"
                                    ]
                            ).format(
                                    eta = eta_string,
                                    iter = iteration,
                                    total_itr = len(dataloader),
                                    meters = str(meters),
                                    lr = optimizer.param_groups[0]["lr"],
                                    GPU_memory = GPU_memory,
                                    memory = memory,
                            ),
                            show_one = True
                    )
                    self.logger.plot_record(value = GPU_memory, win_name = 'GPU')
                    self.logger.plot_record(value = meters.loss.median,
                                            win_name = 'classifier loss,itr:{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(value = memory, win_name = 'memory')
                    self.logger.plot_record(value = optimizer.param_groups[0]["lr"], win_name = 'lr')

                if (total_itr == stop_itr):
                    train_over_flag = True
                    break
                #"""
                if (total_itr % self.cfg.classifier.eval_interval == 0):
                    #synchronize()
                    res_dict = self.evaler_on_all(self.model)
                    f1_micro = res_dict['f1_micro']
                    f1_macro = res_dict['f1_macro']
                    acc = res_dict['acc']
                    self.logger.plot_record(f1_micro,
                                            win_name = 'cls on all micro itr_{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(f1_macro,
                                            win_name = 'cls on all macro itr_{}'.format(ITR),
                                            X_value = total_itr)
                    self.logger.plot_record(acc, win_name = 'cls res ACC itr_{}'.format(ITR),
                                            X_value = total_itr)

                    #synchronize()
                #"""

            if (train_over_flag):
                break
        #"""
        #synchronize()

        self.logger.visdom_text(text = 'start evaling last', win_name = 'state')
        res_dict = self.evaler_on_all(self.model)
        f1_micro = res_dict['f1_micro']
        f1_macro = res_dict['f1_macro']
        acc = res_dict['acc']
        self.checkpointer.save_to_checkpoint_file_with_name(model = self.model,
                                                            filename = 'itr_{}'.format(ITR))
        self.logger.plot_record(f1_micro, win_name = 'cls on all micro itr_{}'.format(ITR),
                                X_value = total_itr)
        self.logger.plot_record(f1_macro, win_name = 'cls on all macro itr_{}'.format(ITR),
                                X_value = total_itr)
        self.logger.plot_record(acc, win_name = 'cls res ACC itr_{}'.format(ITR),
                                X_value = total_itr)

        return res_dict['sentences'], res_dict['preds']

    # def do_label_sentences(self, sentences):
    #     self.logger.info('start do_label_sentences'.format(self.rank))
    #     if (self.model is None):
    #         self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
    #                                                                          num_labels = self.cfg.model.number_classes,
    #                                                                          gradient_checkpointing = True)
    #     self.checkpointer.load_from_filename(self.model, filename = 'longformer', strict = True)
    #     self.model = self.model.to(device)
    #     if (self.distributed):
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
    #         self.model = torch.nn.parallel.DistributedDataParallel(
    #                 model, device_ids = [self.rank], output_device = self.rank,  # find_unused_parameters=True
    #         )
    #     self.model.eval()
    #     # self.logger.info('eval to check before do_label_sentences')
    #     # res = self.evaler(self.model)
    #     # self.logger.info('load from best model, model res:{}'.format(res))
    #     self.logger.info('do_label_sentences total unlabeled sentences:{}'.format(self.rank, len(sentences)))
    #     dataloader_sentence = self.__build_dataloader__(sentences, labels = None, for_train = False)
    #     predicter = Predicter(cfg = self.cfg, logger = self.logger, distributed = True, rank = self.rank,
    #                           dataloader_sentence = dataloader_sentence, model = self.model)
    #     sentences_all, labels = predicter()
    #     self.model.train()
    #     return sentences_all, labels
