import collections


class DotDict(dict):
    """
    A dictionary supporting dot notation.
    From https://gist.github.com/miku/dc6d06ed894bc23dfd5a364b7def5ed8#file-23689767-py
    """
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = DotDict(v)

    def __getattr__(self, item):
        attribute = self.get(item)
        if attribute is None:
            raise AttributeError(f"Attribute {item} not found")
        return attribute

def invert_dict(d):
    return {
        v: k
        for (k, v) in d.items()
    }

def test():
    mydict = {'val': 'it works'}
    nested_dict = {'val': {"a": 'nested works too'}}
    mydict = DotDict(mydict)
    print(mydict.val)
    # 'it works'

    mydict = DotDict(nested_dict)
    print(mydict.val)
    # print(mydict.valsdfsd)
    # 'nested works too'

if __name__ == "__main__":
    test()