"""
@Author: jinzhuan
@File: ace2005.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.utils import load_json


class ACE2005TriggerLoader(Loader):
    """
    The ace2005 dataset processing follows https://github.com/nlpcl-lab/ace2005-preprocessing
    """

    def __init__(self):
        super().__init__()
        self.label_set.add('O')

    def _load(self, path):
        data = load_json(path)
        for item in data:
            for event_mention in item['golden-event-mentions']:
                for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                    trigger_type = event_mention['event_type']
                    if i == event_mention['trigger']['start']:
                        self.label_set.add('B-{}'.format(trigger_type))
                    else:
                        self.label_set.add('I-{}'.format(trigger_type))
        return data

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)


class ACE2005Loader:
    """
    The ace2005 dataset processing follows https://github.com/nlpcl-lab/ace2005-preprocessing
    """

    def __init__(self):
        self.trigger_label_set = set()
        self.trigger_label_set.add('O')
        self.argument_label_set = set()

    def _load(self, path):
        data = load_json(path)
        for item in data:
            for event_mention in item['golden-event-mentions']:
                for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                    trigger_type = event_mention['event_type']
                    if i == event_mention['trigger']['start']:
                        self.trigger_label_set.add('B-{}'.format(trigger_type))
                    else:
                        self.trigger_label_set.add('I-{}'.format(trigger_type))
                """
                28 argument roles

                There are 35 roles in ACE2005 dataset, but the time-related 8 roles were replaced by 'Time' as the previous work (Yang et al., 2016).
                ['Time-At-End','Time-Before','Time-At-Beginning','Time-Ending', 'Time-Holds', 'Time-After','Time-Starting', 'Time-Within'] --> 'Time'.
                """
                for argument in event_mention['arguments']:
                    role = argument['role']
                    if role.startswith('Time'):
                        role = role.split('-')[0]
                    self.argument_label_set.add(role)
        return data

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)

    def get_trigger_labels(self):
        labels = list(self.trigger_label_set)
        labels.sort()
        return labels

    def get_argument_labels(self):
        labels = list(self.argument_label_set)
        labels.sort()
        return labels
