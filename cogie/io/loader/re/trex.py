"""
@Author: jinzhuan
@File: trex.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.utils import load_json
import nltk


class TrexRelationLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = []
        datas = load_json(path)
        count = 0
        for data in datas:
            text = data['text']
            entities = data['entities']
            triples = data['triples']
            sentences_boundaries = data['sentences_boundaries']
            for sentences_boundary in sentences_boundaries:
                entity_mentions = []
                relation_mentions = []
                sentence = text[sentences_boundary[0]:sentences_boundary[1]]
                words = nltk.word_tokenize(sentence)

                for entity in entities:
                    if entity['boundaries'][0] >= sentences_boundary[0] and entity['boundaries'][1] <= \
                            sentences_boundary[1]:
                        entity_mention_position = get_mention_position(text, sentences_boundary, entity['boundaries'])
                        if entity_mention_position[0] >= entity_mention_position[1]:
                            count += 1
                            continue
                        entity_mention = {'position': entity_mention_position}
                        entity_mentions.append(entity_mention)
                for triple in triples:
                    sentence_id = triple['sentence_id']
                    predicate = triple['predicate']
                    subject = triple['subject']
                    object = triple['object']
                    if not subject['boundaries'] or not object['boundaries'] or sentences_boundaries[
                        sentence_id] != sentences_boundary:
                        continue
                    relation_type = predicate['uri']
                    self.label_set.add(relation_type)
                    subject_mention_position = get_mention_position(text, sentences_boundary, subject['boundaries'])
                    object_mention_position = get_mention_position(text, sentences_boundary, object['boundaries'])
                    if subject_mention_position[0] >= subject_mention_position[1]:
                        count += 1
                        continue
                    if object_mention_position[0] >= object_mention_position[1]:
                        count += 1
                        continue
                    arguments = [subject_mention_position, object_mention_position]
                    relation_mention = {'relation_type': relation_type, 'arguments': arguments}
                    relation_mentions.append(relation_mention)
                dataset.append({'sentence': sentence, 'words': words, 'entity_mentions': entity_mentions,
                                'relation_mentions': relation_mentions})
        return dataset

    def load_one(self, path):
        dataset = load_json(path)
        return dataset

    def load_all(self, path):
        datasets = []
        for f in os.listdir(path):
            dataset = self._load(os.path.join(path, f))
            datasets.extend(dataset)
        return datasets


def get_mention_position(text, sentence_boundary, entity_boundary):
    left_text = text[sentence_boundary[0]:entity_boundary[0]]
    right_text = text[sentence_boundary[0]:entity_boundary[1]]
    left_length = len(nltk.word_tokenize(left_text))
    right_length = len(nltk.word_tokenize(right_text))
    return [left_length, right_length]
