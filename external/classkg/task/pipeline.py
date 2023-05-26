import os
import sys
import argparse
import torch

sys.path.append('..')

from torch.multiprocessing import spawn

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '11877'
world_size = 1
device = torch.device('cuda')

from Models.Base.build_classifier import build_classifier_with_cfg
from keyword_sentence.updater_top import Keywords_Updater_TFIDF
from Models.Graph_SSL.trainer_gcn import Trainer_GCN

from PROJECT_ROOT import ROOT_DIR
from compent.logger import Logger
from config import cfg
from keyword_sentence.keywords import KeyWords
from keyword_sentence.sentence import Sentence_ALL
from compent.set_multi_GPUs import set_multi_GPUs_envs
from compent.vote import Vote_All_Unlabeled
from compent.comm import broadcast_data
from compent.utils import set_seed_all
from compent.saver import Saver



def main(rank, cfg_file, visdom_env_name, TOTAL_ITR, seed, lm, clustering, test_mode):
    set_seed_all(seed)
    set_multi_GPUs_envs(rank, world_size)
    cfg_file_path = os.path.join(ROOT_DIR, 'config', cfg_file)
    cfg.merge_from_file(cfg_file_path)

    logger = Logger(name = visdom_env_name, save_dir = cfg.file_path.log_dir, distributed_rank = 0,
                    only_main_rank = False, visdom_port = 8888)
    logger.visdom_text(text = str(cfg), win_name = 'cfg')

    sentence_all = Sentence_ALL(cfg)

    if test_mode:
        classifier = build_classifier_with_cfg(cfg)
        classifier_trainer = classifier(lm, cfg, logger, distributed=True, sentences_all=sentence_all)
        retval = os.getcwd()
        INFERENCE_PATH = os.path.join(retval, '..', 'inference')
        inference_path = os.path.join(INFERENCE_PATH, visdom_env_name)
        os.system(f"mkdir -p {inference_path}")
        classifier_trainer.test_model(f"itr_{TOTAL_ITR-1}", inference_path)
        exit(0)

    keywords = KeyWords(cfg = cfg, logger = logger)

    keywords.analyse_on_GTunlabel(sentence_all)

    vote_all_unlabeled = Vote_All_Unlabeled(cfg, logger, keywords, need_no_confuse = False,
                                            need_static_hit_words = True)

    GCN_trainer = Trainer_GCN(cfg = cfg, logger = logger, distributed = True, sentences_all = sentence_all,
                              keywords = keywords)
    classifier = build_classifier_with_cfg(cfg)
    classifier_trainer = classifier(lm, cfg, logger, distributed = True, sentences_all = sentence_all)
    updater = Keywords_Updater_TFIDF(keywords = keywords, cfg = cfg, logger = logger)

    saver = Saver(save_dir = cfg.file_path.save_dir)

    for cur_itr in range(TOTAL_ITR):
        logger.set_value(key = 'ITR', value = cur_itr)
        logger.info('iteration:{}, start'.format(cur_itr))
        logger.visdom_text(text = 'start ITR:{}'.format(cur_itr), win_name = 'state')

        voted_sentences, voted_label, GT_labels = vote_all_unlabeled(sentence_all.unlabeled_sentence,
                                                                     GT_label = sentence_all.unlabeled_GT_label)

        logger.visdom_text(text = 'start training GIN', win_name = 'state')
        res_dict = GCN_trainer.train_model(sentences = voted_sentences,
                                           vote_labels = voted_label,
                                           GT_labels = GT_labels, ITR = cur_itr, clustering = clustering)

        logger.visdom_text(text = 'start training classifier', win_name = 'state')
        sentences, labels = classifier_trainer.train_model(sentences = res_dict['sentences'],
                                                           labels = res_dict['pred_labels'],
                                                           finetune_from_pretrain = True, ITR = cur_itr)

        saver.save_to_file(obj = [sentences, labels], filename = 'sentence_label_itr_{}'.format(cur_itr))

        logger.visdom_text(text = 'start update keywords', win_name = 'state')
        diff = updater.update_keywords(sentences = sentences, labels = labels, incremental = False)
        logger.plot_record(value = diff, win_name = 'keywords_diff')
        keywords.analyse_on_GTunlabel(sentence_all)

        keywords.dump_keyworks('keywords_{}'.format(cur_itr))
        logger.info('----------------------------------------------------------------------\n\n\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ag_news", help="dataset name")
    parser.add_argument("--total_iter", type=int, default=3)
    parser.add_argument("--lm", type=str, default="bert-base-uncased")
    parser.add_argument("--cluster", action="store_true", default=False)
    parser.add_argument("--test_mode", action="store_true", default=False)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    print(args)
    cfg_file = args.dataset + ".yaml"
    visdom_env_name = args.dataset
    torch.multiprocessing.set_start_method('spawn')
    spawn(main, args = (cfg_file, visdom_env_name, args.total_iter, args.random_state, args.lm, args.cluster, args.test_mode),
          nprocs = world_size, join = True)
    print('finish')