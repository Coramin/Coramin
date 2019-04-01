from collections import abc
from collections import OrderedDict


class OrderedSet(abc.MutableSet):
    def __init__(self, iterable=None):
        self._data = OrderedDict()
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[value] = None

    def discard(self, value):
        self._data.pop(value, None)

    def update(self, iterable):
        for i in iterable:
            self.add(i)

    def intersection(self, iterable):
        ret = OrderedSet()
        for i in iterable:
            if i in self:
                ret.add(i)
        return ret

    def __repr__(self):
        s = '{'
        for i in self:
            s += str(i)
            s += ', '
        s += '}'
        return s

    def __str__(self):
        return self.__repr__()


class OrderedIDDict(abc.MutableMapping):
    def __init__(self, mapping=None):
        self._data = OrderedDict()
        self._keys = OrderedDict()
        if mapping is not None:
            self.update(mapping)

    def __getitem__(self, key):
        return self._data[id(key)]

    def __setitem__(self, key, value):
        self._data[id(key)] = value
        self._keys[id(key)] = key

    def __delitem__(self, key):
        del self._data[id(key)]
        del self._keys[id(key)]

    def __iter__(self):
        return iter(self._keys.values())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        s = '{'
        for i, key in self._keys.items():
            val = self._data[i]
            s += str(key)
            s += ': '
            s += str(val)
            s += ', '
        s += '}'
        return s

    def __str__(self):
        return self.__repr__()


class OrderedIDSet(abc.MutableSet):
    def __init__(self, iterable=None):
        self._data = OrderedIDDict()
        if iterable is not None:
            self.update(iterable)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def add(self, value):
        self._data[value] = None

    def discard(self, value):
        self._data.pop(value, None)

    def update(self, iterable):
        for i in iterable:
            self.add(i)

    def __repr__(self):
        s = '{'
        for i in self:
            s += str(i)
            s += ', '
        s += '}'
        return s

    def __str__(self):
        return self.__repr__()
