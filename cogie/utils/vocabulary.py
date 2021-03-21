"""
@Author: jinzhuan
@File: vocabulary.py
@Desc: 
"""
import io

from collections import Counter
from functools import wraps


class Option(dict):
    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.pop(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


class VocabularyOption(Option):
    def __init__(self,
                 max_size=None,
                 min_freq=None,
                 padding='<pad>',
                 unknown='<unk>'):
        super().__init__(
            max_size=max_size,
            min_freq=min_freq,
            padding=padding,
            unknown=unknown
        )


def _check_build_vocab(func):
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self._word2idx is None or self.rebuild is True:
            self.build_vocab()
        return func(self, *args, **kwargs)

    return _wrapper


def _is_iterable(value):
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


def _check_build_status(func):
    @wraps(func)  # to solve missing docstring
    def _wrapper(self, *args, **kwargs):
        if self.rebuild is False:
            self.rebuild = True
            if self.max_size is not None and len(self.word_count) >= self.max_size:
                pass
                # logger.info("[Warning] Vocabulary has reached the max size {} when calling {} method. "
                #             "Adding more words may cause unexpected behaviour of Vocabulary. ".format(
                #     self.max_size, func.__name__))
        return func(self, *args, **kwargs)

    return _wrapper


class Vocabulary(object):
    def __init__(self, max_size=None, min_freq=None, padding='<pad>', unknown='<unk>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word = Counter()

    @property
    @_check_build_vocab
    def word2idx(self):
        return self._word2idx

    @word2idx.setter
    def word2idx(self, value):
        self._word2idx = value

    @property
    @_check_build_vocab
    def idx2word(self):
        return self._idx2word

    @idx2word.setter
    def idx2word(self, value):
        self._word2idx = value

    @_check_build_status
    def update(self, word_lst, no_create_entry=False):
        self._add_no_create_entry(word_lst, no_create_entry)
        self.word_count.update(word_lst)
        return self

    @_check_build_status
    def add(self, word, no_create_entry=False):
        self._add_no_create_entry(word, no_create_entry)
        self.word_count[word] += 1
        return self

    def _add_no_create_entry(self, word, no_create_entry):
        if isinstance(word, str) or not _is_iterable(word):
            word = [word]
        for w in word:
            if no_create_entry and self.word_count.get(w, 0) == self._no_create_word.get(w, 0):
                self._no_create_word[w] += 1
            elif not no_create_entry and w in self._no_create_word:
                self._no_create_word.pop(w)

    @_check_build_status
    def add_word(self, word, no_create_entry=False):
        self.add(word, no_create_entry=no_create_entry)

    @_check_build_status
    def add_word_lst(self, word_lst, no_create_entry=False):
        self.update(word_lst, no_create_entry=no_create_entry)
        return self

    def build_vocab(self):
        if self._word2idx is None:
            self._word2idx = {}
            if self.padding is not None:
                self._word2idx[self.padding] = len(self._word2idx)
            if (self.unknown is not None) and (self.unknown != self.padding):
                self._word2idx[self.unknown] = len(self._word2idx)

        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self._word2idx is not None:
            words = filter(lambda kv: kv[0] not in self._word2idx, words)
        start_idx = len(self._word2idx)
        self._word2idx.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False
        return self

    def build_reverse_vocab(self):
        self._idx2word = {i: w for w, i in self._word2idx.items()}
        return self

    @_check_build_vocab
    def __len__(self):
        return len(self._word2idx)

    @_check_build_vocab
    def __contains__(self, item):
        return item in self._word2idx

    def has_word(self, w):
        return self.__contains__(w)

    @_check_build_vocab
    def __getitem__(self, w):
        if w in self._word2idx:
            return self._word2idx[w]
        if self.unknown is not None:
            return self._word2idx[self.unknown]
        else:
            raise ValueError("word `{}` not in vocabulary".format(w))

    @property
    def _no_create_word_length(self):
        return len(self._no_create_word)

    def _is_word_no_create_entry(self, word):
        return word in self._no_create_word

    def to_index(self, w):
        return self.__getitem__(w)

    @property
    @_check_build_vocab
    def unknown_idx(self):
        if self.unknown is None:
            return None
        return self._word2idx[self.unknown]

    @property
    @_check_build_vocab
    def padding_idx(self):
        if self.padding is None:
            return None
        return self._word2idx[self.padding]

    @_check_build_vocab
    def to_word(self, idx):
        return self._idx2word[idx]

    def clear(self):
        self.word_count.clear()
        self._word2idx = None
        self._idx2word = None
        self.rebuild = True
        self._no_create_word.clear()
        return self

    def __getstate__(self):
        len(self)  # make sure vocab has been built
        state = self.__dict__.copy()
        # no need to pickle _idx2word as it can be constructed from _word2idx
        del state['_idx2word']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_reverse_vocab()

    def __repr__(self):
        return "Vocabulary({}...)".format(list(self.word_count.keys())[:5])

    @_check_build_vocab
    def __iter__(self):
        # 依次(word1, 0), (word1, 1)
        for index in range(len(self._word2idx)):
            yield self.to_word(index), index

    def save(self, filepath):
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'w', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `filepath`.")

        f.write(f'max_size\t{self.max_size}\n')
        f.write(f'min_freq\t{self.min_freq}\n')
        f.write(f'unknown\t{self.unknown}\n')
        f.write(f'padding\t{self.padding}\n')
        f.write(f'rebuild\t{self.rebuild}\n')
        f.write('\n')
        idx = -2
        for word, count in self.word_count.items():
            if self._word2idx is not None:
                idx = self._word2idx.get(word, -1)
            is_no_create_entry = int(self._is_word_no_create_entry(word))
            f.write(f'{word}\t{count}\t{idx}\t{is_no_create_entry}\n')
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()

    @staticmethod
    def load(filepath):
        if isinstance(filepath, io.IOBase):
            assert filepath.writable()
            f = filepath
        elif isinstance(filepath, str):
            try:
                f = open(filepath, 'r', encoding='utf-8')
            except Exception as e:
                raise e
        else:
            raise TypeError("Illegal `filepath`.")

        vocab = Vocabulary()
        for line in f:
            line = line.strip('\n')
            if line:
                name, value = line.split()
                if name in ('max_size', 'min_freq'):
                    value = int(value) if value != 'None' else None
                    setattr(vocab, name, value)
                elif name in ('unknown', 'padding'):
                    value = value if value != 'None' else None
                    setattr(vocab, name, value)
                elif name == 'rebuild':
                    vocab.rebuild = True if value == 'True' else False
            else:
                break
        word_counter = {}
        no_create_entry_counter = {}
        word2idx = {}
        for line in f:
            line = line.strip('\n')
            if line:
                parts = line.split('\t')
                word, count, idx, no_create_entry = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                if idx >= 0:
                    word2idx[word] = idx
                word_counter[word] = count
                if no_create_entry:
                    no_create_entry_counter[word] = count

        word_counter = Counter(word_counter)
        no_create_entry_counter = Counter(no_create_entry_counter)
        if len(word2idx) > 0:
            if vocab.padding:
                word2idx[vocab.padding] = 0
            if vocab.unknown:
                word2idx[vocab.unknown] = 1 if vocab.padding else 0
            idx2word = {value: key for key, value in word2idx.items()}

        vocab.word_count = word_counter
        vocab._no_create_word = no_create_entry_counter
        if word2idx:
            vocab._word2idx = word2idx
            vocab._idx2word = idx2word
        if isinstance(filepath, str):  # 如果是file的话就关闭
            f.close()
        return vocab
