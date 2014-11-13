from uuid import uuid4


def isolate_namespace(name):
    return '_a%s%s' % (uuid4().hex, name)


def is_dunder(name):
    return name.startswith('__') and name.endswith('__')
