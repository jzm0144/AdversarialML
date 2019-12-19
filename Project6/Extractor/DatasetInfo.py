import os
import string
from typing import Optional, Union

import pickledb

UNIGRAM_FEATURE_SET = "ncu"
STYLOMETRY_FEATURE_SET = "sty"
BOW_FEATURE_SET = "bow"
CHARACTER_GRAM = "char-gram"

FS_MAP = {
    UNIGRAM_FEATURE_SET: 'Unigram',
    STYLOMETRY_FEATURE_SET: 'Stylometry',
    BOW_FEATURE_SET: 'Bag-of-Words',
    CHARACTER_GRAM: 'N-Gram',
}

FS_MAP_REVERSE = {FS_MAP[key]: key for key in FS_MAP}

FEATURE_SET_KEY = "feature_sets"

DATASET_INFO_DEFAULT = {
    'info': {
        'data_dir': ''
    },
    "authors": {},
    "instances": {},
    "feature_sets": {},
    "test": {},  # dir
    'validation': {}  # dir
}

DATASET_FS_INFO_DEFAULT = {
    "name": "",
    "size": -1
}

database_dir = ""

SET_INFORMATION = [
    {
        'name': 'test',
        'ext': '.test',
        'folder_ext': '_test'
    },
    {
        'name': 'validation',
        'ext': '.val',
        'folder_ext': '_val'
    },
    # ...
]


def SingleLevelDataset(path, descriptor=None):
    result = {}
    instances = os.listdir(database_dir + path)
    for instance in instances:
        if instance == "test" or os.path.isdir(instance):
            continue

        result[instance] = {
            'path': os.path.join(database_dir, path, instance)
        }

    return result


def AuthorLevelDataset(path, descriptor=None):
    authors = os.listdir(database_dir + path)
    result = {}
    for author in authors:
        author_instances = os.listdir(database_dir + path + "/" + author)
        for instance in author_instances:
            result[instance] = {
                'path': os.path.join(database_dir, path, author, instance),
                'author': author
            }
    return result


class DatasetDescriptor:
    SingleLevel = "single"
    AuthorLevel = "author"

    def __init__(self, level=None):
        self.level = level

    def setLevel(self, level):
        self.level = level

    def loadLevel(self, data_dir):
        files = os.listdir(database_dir + data_dir)
        counter = 0
        for file in files:
            if file == "test":
                continue
            if os.path.isdir(database_dir + data_dir + file):
                counter += 1
                if counter > 2:
                    break

        if counter > 2:
            level = DatasetDescriptor.AuthorLevel
        else:
            level = DatasetDescriptor.SingleLevel

        self.level = level

    def getFiles(self, data_dir):
        if self.level == DatasetDescriptor.AuthorLevel:
            return AuthorLevelDataset(data_dir, descriptor=self)
        else:
            return SingleLevelDataset(data_dir, descriptor=self)

    def getAuthor(self, instance):
        return None

    def get_test(self):
        return []


class CASISDataset(DatasetDescriptor):
    def __init__(self):
        super().__init__()

    def getAuthor(self, instance):
        return instance.split("_")[0]


class DatasetInfo:
    name: string
    feature_set: string
    _data: Optional[pickledb.PickleDB]
    descriptor: Optional[DatasetDescriptor]
    DATASET = 0
    FEATURE_SET = 1

    def __init__(self, name, is_dir=False, descriptor="auto", relative_dir="./"):
        self.relative_dir = relative_dir
        if descriptor == "auto":
            descriptor = DatasetInfo.auto_descriptor(name)
        self.descriptor = descriptor
        self.dir = is_dir
        self.name = name
        self.feature_set = ""
        self.fs_pointer = None
        self._data = None
        self.dirty = False

    @staticmethod
    def auto_descriptor(name):
        descriptor = DatasetDescriptor
        if name.split("_")[0].lower().find('casis') != -1:
            descriptor = CASISDataset
        return descriptor()

    @property
    def data(self):
        return self._data

    @property
    def set_info(self):
        return self._data.get('info')

    @property
    def test(self):
        return self._data.get("test")

    def set_data(self, val):
        self._data = val
        self.dirty = True

        return self

    def path(self):
        dir_ext = ""

        parts = self.name.split("_")
        name = parts[0]  # "_".join([x for i, x in enumerate(parts) if i != len(parts) - 1])

        return self.relative_dir+"datasets/" + name + dir_ext + ".info"

    def add_instance(self, name, author=None, test_instance=False):
        if self._data is None:
            return False

        if author is None and self.descriptor is not None:
            author = self.descriptor.getAuthor(name)

        if author is not None:
            if not test_instance:
                if not self._data.dexists("authors", author):
                    self._data.dadd("authors", [author, []])

                data = self._data.dget("authors", author)
                if name not in data:
                    data.append(name)
            else:
                test = self._data.get("test")
                if not isinstance(test, dict):
                    test = {'instances': {}}
                    self._data.set('test', test)
                elif 'instances' not in test:
                    test['instances'] = {}
                test['instances'][name] = author

        if not test_instance:
            self._data.dadd("instances", [name, author])

        self.dirty = True
        return self

    @property
    def instances(self) -> Union[bool, dict]:
        if self._data is None:
            return False

        return self._data.get("instances")

    def get_instances_list(self):
        instances = self.instances
        if instances:
            instances = list(instances.keys())
        return instances

    @property
    def authors(self) -> Union[bool, dict]:
        if self._data is None:
            return False

        return self._data.get("authors")

    def get_authors_list(self):
        authors = self.authors
        if authors:
            authors = list(authors.keys())
        return authors

    def set_feature_name(self, feature_set):
        if self._data is None:
            return False

        self.feature_set = feature_set

        splits = feature_set.split("_")

        if not self._data.dexists(FEATURE_SET_KEY, splits[0]):  # self.feature_set):
            data = {}
            data["name"] = FS_MAP.get(splits[0], None)
            self._data.dadd(FEATURE_SET_KEY, [splits[0], data])

        current = self._data.dget(FEATURE_SET_KEY, splits[0])
        for split_index, split in enumerate(splits):
            if split_index == len(splits)-1:
                default = DATASET_FS_INFO_DEFAULT.copy()
                for key in default:
                    if key not in current:
                        current[key] = default[key]
            else:
                if split not in current:
                    next = splits[split_index+1]
                    if next not in current:
                        current[next] = {}
                    current = current.get(next)

        self.fs_pointer = current
        self.save()
        return self

    def set_feature_prop(self, prop, value):
        if self._data is None:
            return False

        self.fs_pointer[prop] = value

        return self

    def get_features(self):
        if self._data is None:
            return False

        return self.fs_pointer.copy()

    def get_feature_prop(self, prop, default=None):
        if self._data is None:
            return False
        if not self._data.dexists(FEATURE_SET_KEY, self.feature_set):
            return {}
        feature_set_info = self._data.dget(FEATURE_SET_KEY, self.feature_set)
        return feature_set_info.get(prop, default)

    def test_set_exists(self) -> bool:
        if self._data is None:
            return False

        test = self._data.get("test")
        if test is None or not isinstance(test, dict):
            return False

        instances = test.get("instances")
        if instances is None or len(instances) == 0:
            return False

        return True

    def get_fs_from_name(self):
        splits = self.name.split('_')
        if len(splits) < 2:
            return ''
        return "_".join(splits[1:])

    def read(self):
        file = self.path()
        create = False
        if not os.path.exists(file):
            create = True

        self._data = pickledb.load(file, False)

        if create:
            self.dirty = True
            for key in DATASET_INFO_DEFAULT:
                self._data.set(key, DATASET_INFO_DEFAULT[key])
        else:
            for key in DATASET_INFO_DEFAULT:
                if not self._data.exists(key):
                    self._data.set(key, DATASET_INFO_DEFAULT[key])
                    self.dirty = True

        feature_set = self.get_fs_from_name()
        self.set_feature_name(feature_set)

        return self

    def save(self, force=False):
        if self._data is not None:
            if force is True or self.dirty is True:
                self._data.dump()

        return self
