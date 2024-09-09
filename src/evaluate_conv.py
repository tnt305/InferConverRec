import json
import re
from collections import Counter

import rouge
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

from utils import normalize_answer

# year_pattern = re.compile(r'\(\d{4}\)')
slot_pattern = re.compile(r'<movie>')


class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer

        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1)
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                         decoded_preds]
        decoded_preds = [normalize_answer(pred.strip()) for pred in decoded_preds]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
                          decoded_labels]
        decoded_labels = [normalize_answer(label.strip()) for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_item_ratio(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.compute_rouge(decoded_preds, decoded_labels)
        self.compute_intra_dist(decoded_preds)
        self.sent_cnt += len(
            [pred for pred in decoded_preds if len(pred) > 0]
        )  # number of samples: rouge, intra-distinct

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].append(token)

    def compute_intra_dist(self, preds):
        for k in range(1, 5):
            dist_k = f'intra-dist@{k}'
            self.metric[dist_k] += self._intra_dist(preds, k)

    def _intra_dist(self, preds, k):
        intra = 0.0
        for pred in preds:
            pred = pred.split()
            if len(pred) == 0:
                continue
            counts = Counter(ngrams(pred, k))
            intra += max(len(counts), 1e-12) / max(sum(counts.values()), 1e-5)
        return intra

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(1, 5):
                weights = [1 / k for _ in range(k)]  # Correct weights
                self.metric[f'bleu@{k}'] += sentence_bleu(label, pred, weights=weights)

    def compute_item_ratio(self, strs):
        for str in strs:
            # items = re.findall(year_pattern, str)
            # self.metric['item_ratio'] += len(items)
            items = re.findall(slot_pattern, str)
            self.metric['item_ratio'] += len(items)

    def compute_rouge(self, preds, labels):
        for pred, label in zip(preds, labels):
            label = [label]
            result = self._rouge(pred, label)
            self.metric['rouge@1'] += result[0]
            self.metric['rouge@2'] += result[1]
            self.metric['rouge@l'] += result[2]

    def _rouge(self, guess, answers, measure='r'):
        """Compute ROUGE score."""
        evaluator = rouge.Rouge(
            metrics=['rouge-n', 'rouge-l'], max_n=2
        )
        score = evaluator.get_scores(guess, answers)
        return [score['rouge-1'][measure], score['rouge-2'][measure], score['rouge-l'][measure]]

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            elif ('rouge' in k) or ('intra-dist' in k):
                report[k] = v / self.sent_cnt
            elif 'dist' in k:
                report[k] = len(set(v)) / len(v)
            else:
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'rouge@1': 0,
            'rouge@2': 0,
            'rouge@l': 0,
            'intra-dist@1': 0,
            'intra-dist@2': 0,
            'intra-dist@3': 0,
            'intra-dist@4': 0,
            'dist@1': list(),
            'dist@2': list(),
            'dist@3': list(),
            'dist@4': list(),
            'item_ratio': 0,
        }
        self.sent_cnt = 0
