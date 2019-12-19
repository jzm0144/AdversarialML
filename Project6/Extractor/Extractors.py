import os
import pickle
import string
import re

import numpy as np
from nltk.tokenize import sent_tokenize
from collections import Counter, OrderedDict
from abc import abstractmethod, ABC

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, Normalizer

import Data_Utils
import Extractor.DatasetInfo as DatasetInfo

import logging

logging.getLogger()

compound_keywords = ["for", "and", "nor", "but", "or", "yet", "so", "however", "moreover", "nevertheless",
                     "nonetheless", "therefore"]
complex_keywords = ["because", "since", "so", "that", "although", "even", "though", "whereas", "while", "where",
                    "wherever", "how", "however", "if", "whether", "unless", "that", "which", "who", "whom", "after",
                    "as", "before", "since", "when", "whenever", "while", "until"]

database_dir = ""

WRITE_TO_FILE = 0
WRITE_TO_MEMORY = 1


class Extractor:
    def __init__(self, data_dir, name, test_dir=None, out_file=None, descriptor="auto", is_dir=False,
                 dataset_level="auto"):
        self.descriptor = descriptor
        self.is_dir = is_dir
        self.dataset_level = dataset_level
        self.name = name

        self._out_file = out_file

        self.feature_size = -1
        if 'feature_name' not in self.__dict__:
            self.feature_name = "dummy"
        self.data_dir = data_dir
        self.test_dir = test_dir

        self._data = {'features': {}, 'lookup_table': {}}
        self._preprocessors = []
        self.params = [[], []]
        self.name_ext = ""

        self.write_type = WRITE_TO_FILE

    def start(self):
        logging.info("Extractor Started")

        if len(self.params[1]) > 0:
            logging.info("Processing parameters")
            defaults = self.params[0].__init__.__defaults__[1:1 + len(self.params[1])]
            ext = ""
            for i, param in enumerate(self.params[1]):
                if param not in self.__dict__ or self.__dict__[param] == defaults[i]:
                    continue

                if ext != "":
                    ext += "-"

                ext += param.replace("_", "-") + "=" + str(self.__dict__[param])
            self.name_ext = ext

        logging.info("DatasetInfo is loading")
        self.info = DatasetInfo.DatasetInfo(self.out_file, descriptor=self.descriptor, is_dir=self.is_dir)

        if self.dataset_level == "auto":
            if self.info.descriptor.level is None:
                self.info.descriptor.loadLevel(self.data_dir)
        else:
            self.info.descriptor.setLevel(self.dataset_level)

        self.info.read()
        logging.info("DatasetInfo loaded")

        set_info = self.info.data
        if not set_info.dexists('info', 'data_dir') or set_info.dget('info', 'data_dir') == '':
            set_info.dadd('info', ('data_dir', self.data_dir))

        if self.test_dir is not None and not set_info.dexists('info', 'test_dir'):
            logging.info("Test set detected")
            set_info.dadd('info', ('test_dir', self.test_dir))

        # Pipeline
        logging.info("Extractor pipeline -> prepare")
        self.prepare()
        logging.info("Extractor pipeline -> run")
        self.run()
        logging.info("Extractor pipeline -> post run")
        self.post_run()
        logging.info("Finished")

        if self.write_type == WRITE_TO_MEMORY:
            return self._data['features']
        return ['filepath']

    @property
    def out_file(self):
        if self._out_file is None:
            file_name = self.name + "_" + self.feature_name

            if self.name_ext != "":
                file_name += "_" + self.name_ext

            return file_name

        return self._out_file


    @property
    def lookup_table(self):
        return self._data['lookup_table']

    def get_target_and_extension(self, is_test=False):
        if is_test:
            target_dir = self.test_dir
            extension = ".test"
        else:
            target_dir = self.data_dir
            extension = ".txt"
        return target_dir, extension

    @abstractmethod
    def prepare(self):
        pass

    def post_run(self):
        self.info.save()
        self.preprocess()

    def run(self):
        logging.info("Extractor pipeline -> run -> process")
        self.process()
        logging.info("Extractor pipeline -> run -> after run process")
        self.after_run_process()

        if self.test_dir is not None:
            logging.info("Extractor pipeline -> run -> before test process")
            self.before_test_process()
            logging.info("Extractor pipeline -> run -> process")
            self.process(True)
            logging.info("Extractor pipeline -> run -> after test process")
            self.after_test_process()

        if self.feature_size != -1:
            self.info.set_feature_prop("size", self.feature_size)

    def after_run_process(self):
        return None

    def before_test_process(self):
        pass

    def after_test_process(self):
        pass

    def process(self, is_test=False):
        raise NotImplementedError

    def parse_file(self, file_name, info, is_test=False):
        pass

    def write_result(self, feature_dict, lookup_table=None, is_test=False):
        _, extension = self.get_target_and_extension(is_test)

        if lookup_table is not None:
            self._data['lookup_table'] = lookup_table

        if self.write_type == WRITE_TO_MEMORY:
            self._data['features'][int(is_test)] = feature_dict
            return feature_dict

        buffer = ""
        for file_name in feature_dict:
            feature_string = ','.join(str(v) for v in feature_dict[file_name])
            buffer += "%s,%s\n" % (file_name, feature_string)

        file = open('./datasets/' + self.out_file + extension, 'w')
        file.write(buffer)
        file.close()

    def preprocess(self):
        if len(self._preprocessors) == 0:
            return True
        pass


class Unigram(Extractor):
    def __init__(self, data_dir, name, test_dir=None, out_file=None, descriptor="auto"):
        self.feature_name = DatasetInfo.UNIGRAM_FEATURE_SET
        super().__init__(data_dir, name, test_dir, out_file, descriptor)


    def process(self, is_test=False):
        target_dir, _ = self.get_target_and_extension(is_test)

        files = self.info.descriptor.getFiles(target_dir)

        feature_dict = {}
        for file_name in files:
            features = self.parse_file(file_name, files[file_name], is_test)
            if features is not None:
                feature_dict[file_name.split(".")[0]] = features
                # buffer += (file_name.split(".")[0] + ',' + features + "\n")  # Read contents of the file

        lookup_table = [chr(x+32) for x in range(95)]

        self.write_result(feature_dict, lookup_table=lookup_table, is_test= is_test)
        return files

    def parse_file(self, key, info, is_test=False):
        if not ('path' in info and os.path.exists(info.get('path'))):
            print("Warning path not found. Key = " + str(key) + " Data = ", info)
            return None

        file = open(info.get("path"), "r", errors='ignore')
        self.info.add_instance(key.split(".")[0], info.get("author", None), is_test)
        feature_vector = self.calculate(file.read())
        if self.feature_size == -1:
            self.feature_size = len(feature_vector)
        features = feature_vector
        file.close()
        return features

    @staticmethod
    def calculate(text):
        unigram_list = [0] * 95
        for i, c in enumerate(text):
            index = ord(c) - 32
            if index > 94 or index < 0:
                continue
            unigram_list[index] = unigram_list[index] + 1
        return unigram_list

class CharacterGram(Extractor):
    def __init__(self, data_dir, name, test_dir=None, gram=4, limit=0, out_file=None,
                 descriptor="auto"):
        self.gram = gram
        self.limit = limit
        self.file_dict = {}
        self.test_dict = {}
        self.feature_name = DatasetInfo.CHARACTER_GRAM
        # ext = "gram="+str(self.gram)+"-limit="+str(self.limit)+"-occurrence="+str(self.occurrence)
        super().__init__(data_dir, name, test_dir, out_file, descriptor)

        self.params = [CharacterGram, ["gram", "limit", "occurrence", "unique_occurrence"]]

    def after_run_process(self):
        self.collect_feature_list(self.file_dict, self._data["run_files"])
        self.generate(self.out_file, False)

    def after_test_process(self):
        self.generate(self.out_file, True)

    def find_ngrams(self, input_list):
        return zip(*[input_list[i:] for i in range(self.gram)])

    def process(self, is_test=False):
        if is_test:
            target_dir = self.test_dir
            current_dict = self.test_dict
            data_key = "test_files"
        else:
            target_dir = self.data_dir
            current_dict = self.file_dict
            data_key = "run_files"

        files = self.info.descriptor.getFiles(target_dir)
        for file_name in files:
            data = files[file_name]

            file = open(data['path'], "r", errors='ignore')
            features = file.read().lower()  # Read contents of the file
            file.close()  # We don't need that file anymore.
            self.info.add_instance(file_name.split(".")[0], data.get("author"))

            if data.get('author') is None:
                data['author'] = self.info.descriptor.getAuthor(file_name)

            if features is not None:
                current_dict[file_name] = Counter(self.find_ngrams(features))  # Read contents of the file
            else:
                del files[file_name]
                continue

        self._data[data_key] = files
        return files

    def collect_feature_list(self, file_data: dict, info):
        self.grammar = Counter()
        for key in file_data:
            tokens = file_data[key]
            self.grammar += tokens

        if 0 < self.limit < len(self.grammar.keys()):
            limited = self.grammar.most_common(self.limit)
            new = {}
            for element in limited:
                new[element[0]] = element[1]
            self.grammar = Counter(new)

    def generate(self, out_file, is_test=False):
        if is_test:
            extension = ".test"
            file_data = self.test_dict
        else:
            extension = ".txt"
            file_data = self.file_dict

        buffer = ""
        if not is_test:
            self.feature_size = -1

        fresh_grammar = Counter({x: 0 for x in self.grammar})

        feature_dict = {}
        lookup_table = []
        for key in file_data:
            word_dict = dict(fresh_grammar.copy())
            tokens = file_data[key]
            for token in tokens:
                if token in word_dict:
                    word_dict[token] = tokens[token]
            # output_dict[key] = word_dict
            if not is_test and self.feature_size == -1:
                self.feature_size = len(word_dict.keys())

            feature_dict[key.split('.')[0]] = list(word_dict.values())
            lookup_table = list(word_dict.keys())

        self.write_result(feature_dict, lookup_table=lookup_table, is_test=is_test)


class BagOfWords(Extractor):
    FeatureCounter: {}

    def __init__(self, data_dir, name, test_dir=None, unique_occurrence=0, out_file=None, descriptor="auto"):
        self.is_dir = False
        self.feature_name = DatasetInfo.BOW_FEATURE_SET
        self.FeatureCounter = {}
        self.unique_occurrence = 0
        self.occurrence = 0
        super().__init__(data_dir, name, test_dir=test_dir, out_file=out_file, descriptor=descriptor)
        self.unique_occurrence = unique_occurrence
        self.params = [BagOfWords, ["unique_occurrence"]]
        self.unique_dict = Counter()
        self.file_dict = {}
        self.test_dict = {}

    def collect_feature_list(self, file_data: dict, info):
        self.FeatureCounter = Counter()
        self.unique_grammar = Counter()
        for key in file_data:
            tokens = file_data[key]
            unique_tokens = Counter({x: 1 for x in tokens})

            self.FeatureCounter += tokens
            self.unique_grammar += unique_tokens

        if self.unique_occurrence > 0:
            self.FeatureCounter = {x: self.FeatureCounter[x] for x in self.FeatureCounter
                                   if x in self.unique_grammar and self.unique_grammar[x] >= self.unique_occurrence}

        self.FeatureCounter = dict(sorted(self.FeatureCounter.items()))

    def after_run_process(self):
        self.collect_feature_list(self.file_dict, self._data["run_files"])
        self.generate(self.out_file, False)

    def after_test_process(self):
        self.generate(self.out_file, True)

    def process(self, is_test=False):
        if is_test:
            target_dir = self.test_dir
            current_dict = self.test_dict
            data_key = "test_files"
        else:
            target_dir = self.data_dir
            current_dict = self.file_dict
            data_key = "run_files"

        files = self.info.descriptor.getFiles(target_dir)
        for file_name in files:
            data = files[file_name]

            features = self.parse_file(file_name, data, is_test)

            if features is not None:
                current_dict[file_name] = features  # Read contents of the file
            else:
                del files[file_name]
                continue

        self._data[data_key] = files
        return files

    def parse_file(self, file_name, info, is_test=False):
        if not ('path' in info):
            return None

        if not ('author' in info):
            author = self.info.descriptor.getAuthor(file_name)
            if author is None:
                print("Author is not identified for " + file_name, info, self.info.descriptor.level)
                exit()
            info['author'] = author

        file = open(info['path'], "r", errors='ignore')
        features = self.extract_from_file(file)  # Read contents of the file
        file.close()  # We don't need that file anymore.
        self.info.add_instance(file_name.split(".")[0], info.get("author"), is_test)
        return features

    def extract_from_file(self, file):
        contents = file.read().lower()
        tokens = tokenize(contents)
        counter = Counter()
        for token in tokens:
            if token in counter:
                counter[token] += 1
            else:
                counter[token] = 1

        return counter

    def generate(self, out_file, is_test=False):
        if is_test:
            extension = ".test"
            file_data = self.test_dict
        else:
            extension = ".txt"
            file_data = self.file_dict

        buffer = ""
        if not is_test:
            self.feature_size = -1

        empty_set = Counter({x: 0 for x in self.FeatureCounter})
        feature_dict = {}
        lookup_table = []
        for key in file_data:
            word_dict = dict(empty_set.copy())
            tokens = file_data[key]
            for token in tokens:
                if token in word_dict:
                    word_dict[token] += tokens[token]
            # output_dict[key] = word_dict

            if not is_test and self.feature_size == -1:
                self.feature_size = len(word_dict.keys())

            feature_dict[key.split('.')[0]] = list(word_dict.values())
            lookup_table = list(word_dict.keys())
            # feature_string = ','.join(str(word_dict[v]) for v in word_dict)
            # buffer += ("%s,%s\n" % (key, feature_string))

        self.write_result(feature_dict, lookup_table=lookup_table, is_test=is_test)

def camel(s):
    return s != s.lower() and s != s.upper() and "_" not in s

def get_func_word_freq(words, funct_words):
    out = [0] * len(funct_words)
    for i, fc in enumerate(funct_words):
        out[i] = words.count(fc)

    return out


def tokenize(s):
    tokens = re.split(r"[^0-9A-Za-z\-'_]+", s)
    return tokens


def get_yules(tokens):
    """
    Returns a tuple with Yule's K and Yule's I.
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)
    In production this needs exception handling.
    """
    token_counter = Counter(tokens)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    if m1 == m2:
        i = (m1 * m1)
    else:
        i = (m1 * m1) / (m2 - m1)
    k = 1 / i * 10000
    return k


class Stylomerty(Extractor):
    dir: ""

    def __init__(self, data_dir, name, test_dir=None, out_file=None, descriptor="auto"):
        self.file_dict = {}
        self.test_dict = {}
        super().__init__(data_dir, name, test_dir=test_dir, out_file=out_file, descriptor=descriptor)
        self.feature_name = DatasetInfo.STYLOMETRY_FEATURE_SET

    def process(self, is_test=False):
        if is_test:
            target_dir = self.test_dir
            current_dict = self.test_dict
        else:
            target_dir = self.data_dir
            current_dict = self.file_dict

        files = self.info.descriptor.getFiles(target_dir)
        for file_name in files:
            data = files[file_name]

            file_name_no_ext = file_name.split(".")[0]
            file = open(data['path'], "r", errors='ignore')
            features = file.read().lower()  # Read contents of the file
            file.close()  # We don't need that file anymore.
            self.info.add_instance(file_name_no_ext, data.get("author"), is_test)

            if features is not None:
                current_dict[file_name_no_ext] = features  # Read contents of the file
            else:
                del files[file_name]
                continue

        self.generate(self.out_file, is_test)
        return files

    @staticmethod
    def stylometry(text):
        fv = []
        # note: the nltk.word_tokenize includes punctuation
        words2 = tokenize(text)
        text = text.lower()
        words = tokenize(text)
        sentences = sent_tokenize(text)
        vocab = set(words)
        words_per_sentence = len(words) / len(sentences)

        vocab_richness = [get_yules(words), np.average(words_per_sentence)]

        for i in range(10):
            vocab_richness.append(len([item for item in vocab if words.count(item) == i + 1]))
        fv.append(vocab_richness)

        # lengths (2)
        lengths = [len(words), len(text)]
        fv.append(lengths)

        fv_shape = [0] * 5
        for word in words2:
            if all(c.isupper() for c in word):
                fv_shape[0] += 1
            elif all(c.islower() for c in word):
                fv_shape[1] += 1
            elif word.istitle():
                fv_shape[2] += 1
            elif camel(word):
                fv_shape[3] += 1
            else:
                fv_shape[4] += 1

        fv.append(fv_shape)

        word_length = [0] * 20
        for word in words:
            wlen = len(word)
            if wlen < 21:
                word_length[wlen - 1] += 1
        fv.append(word_length)

        atoz = string.ascii_lowercase
        letter_count = [0] * len(atoz)
        for i, l in enumerate(atoz):
            letter_count[i] = text.count(l)
        fv.append(letter_count)

        digits = '0123456789'
        digit_count = [0] * len(digits)
        for i, d in enumerate(digits):
            digit_count[i] = text.count(d)
        fv.append(digit_count)

        punctuation = '.?!,;:()"-\''
        fv_punct = []
        for char in punctuation:
            fv_punct.append(text.count(char))
        fv.append(fv_punct)

        special_characters = "`‘˜@#$%ˆ&*_+=[]{}\\|/<>"
        fv_special = []
        for char in special_characters:
            fv_special.append(text.count(char))
        fv.append(fv_special)

        function_words = tokenize(
            "a about above after again ago all almost along already also although always am among an and another any "
            "anybody anything anywhere are aren't around as at back else be been before being below beneath beside "
            "between beyond billion billionth both each but by can can't could couldn't did didn't do does doesn't "
            "doing done don't down during eight eighteen eighteenth eighth eightieth eighty either eleven eleventh "
            "enough even ever every everybody everyone everything everywhere except far few fewer fifteen fifteenth "
            "fifth fiftieth fifty first five for fortieth forty four fourteen fourteenth fourth hundred from get gets "
            "getting got had hadn't has hasn't have haven't having he he'd he'll hence her here hers herself he's him "
            "himself his hither how however near hundredth i i'd if i'll i'm in into is i've isn't it its it's itself "
            "last less many me may might million millionth mine more most much must mustn't my myself near nearby "
            "nearly neither never next nine nineteen nineteenth ninetieth ninety ninth no nobody none noone nothing "
            "nor not now nowhere of off often on or once one only other others ought oughtn't our ours ourselves out "
            "over quite rather round second seven seventeen seventeenth seventh seventieth seventy shall shan't she'd "
            "she she'll she's should shouldn't since six sixteen sixteenth sixth sixtieth sixty so some somebody "
            "someone something sometimes somewhere soon still such ten tenth than that that that's the their theirs "
            "them themselves these then thence there therefore they they'd they'll they're third thirteen thirteenth "
            "thirtieth thirty this thither those though thousand thousandth three thrice through thus till to towards "
            "today tomorrow too twelfth twelve twentieth twenty twice two under underneath unless until up us very "
            "when was wasn't we we'd we'll were we're weren't we've what whence where whereas which while whither who "
            "whom whose why will with within without won't would wouldn't yes yesterday yet you your you'd you'll "
            "you're yours yourself yourselves you've")

        function_frequency = get_func_word_freq(text, function_words)
        fv.append(function_frequency)

        return [item for sublist in fv for item in sublist]

    def generate(self, out_file, is_test=False):
        if is_test:
            extension = ".test"
            file_data = self.test_dict
        else:
            extension = ".txt"
            file_data = self.file_dict

        buffer = ""
        if not is_test:
            self.feature_size = -1

        feature_dict = {}
        for key in file_data:
            text = file_data[key]

            feature_vector = self.stylometry(text)
            feature_dict[key] = feature_vector
            if not is_test and self.feature_size == -1:
                self.feature_size = len(feature_vector)

        self.write_result(feature_dict, is_test)

