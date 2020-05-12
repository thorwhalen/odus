from itertools import chain
import types


def add_method(obj, method_func, method_name=None, class_name=None):
    """
    Dynamically add a method to an object.

    :param obj: The object to add a method to
    :param method_func: The function to use as a method. The first argument must be the object itself
        (usually called self)
    :param method_name: The desired function name. If None, will take method_func.__name__
    :param class_name:  The desired class name. If None, will take type(obj).__name__
    :return: the object, but with the additional method (or a different function for it)

    >>> class A:
    ...     def __init__(self, x=10):
    ...         self.x = x
    >>> def times(self, y):
    ...     return self.x * y
    >>> def plus(self, y):
    ...     return self.x + y
    >>> a = A(x=10)
    >>> a = add_method(a, plus, '__call__')  # add a __call__ method, assigning it to plus
    >>> a(2)
    12
    >>> a = add_method(a, times, '__call__')  # reassign the __call__ method to times instead
    >>> a(2)
    20
    >>> a = add_method(a, plus, '__getitem__')  # assign the method __getitem__ to plus
    >>> a[2]  # see that it works
    12
    >>> a(2)  # and that we still have our __call__ method
    20
    """
    if isinstance(method_func, str):
        method_name = method_func
        method_func = getattr(obj, method_name)
    if method_name is None:
        method_name = method_func.__name__

    base = type(obj)

    if class_name is None:
        class_name = base.__name__
    bases = (base.__bases__[1:]) + (base,)
    bases_names = set(map(lambda x: x.__name__, bases))
    if class_name in bases_names:
        for i in range(6):
            class_name += '_'
            if not class_name in bases_names:
                break
        else:
            raise ValueError("can't find a name for class that is not taken by bases. Consider using explicit name")

    new_keys = set(dir(obj)) - set(chain(*[dir(b) for b in bases]))

    d = {a: getattr(obj, a) for a in new_keys}
    d[method_name] = method_func

    return type(class_name, bases, d)()


def inject_method(obj, method_func, method_name=None):
    """
    Inject one or several methods to an object.
    Note that it doesn't work if method_name is a dunder.
    If you want to inject things like __call__, __getitem__, etc., use add_method instead.
    :param obj: The object to add a method to
    :param method_func: The function to use as a method. The first argument must be the object itself
        (usually called self)
    :param method_name: The desired function name. If None, will take method_func.__name__
    :return: the object, but with the additional method (or a different function for it)

    >>> class A:
    ...     def __init__(self, x=10):
    ...         self.x = x
    >>> def times(self, y):
    ...     return self.x * y
    >>> def plus(self, y):
    ...     return self.x + y
    >>> a = A(x=10)
    >>> a = inject_method(a, plus, 'add')  # inject the plus method (but call it 'add')
    >>> a.add(2)
    12
    >>> a = inject_method(a, times)  # inject the times method (and use it's name automatically)
    >>> a.times(2)
    20
    >>> a = inject_method(a, plus, 'times')  # be evil: Inject the plus method but call it 'times'
    >>> a.times(2)
    12
    """
    if isinstance(method_func, types.FunctionType):
        if method_name is None:
            method_name = method_func.__name__
        setattr(obj,
                method_name,
                types.MethodType(method_func, obj))
    else:
        if isinstance(method_func, dict):
            method_func = [(func, func_name) for func_name, func in method_func.items()]
        for method in method_func:
            if isinstance(method, tuple) and len(method) == 2:
                obj = inject_method(obj, method[0], method[1])
            else:
                obj = inject_method(obj, method)

    return obj
