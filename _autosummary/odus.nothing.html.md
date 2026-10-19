# odus.nothing

Universal neutral (identity) elements for accumulation.

Provides objects that act as the neutral element of an operation whatever the
operand type, so accumulators can be initialized without knowing the type (or
shape) of what they will accumulate.

### Functions

| [`get_universal_neutral_for_method`](#odus.nothing.get_universal_neutral_for_method)(method_name)   | A universal neutral element (synonyms: "identity element" and "unit element").   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`get_universal_neutral_for_methods`](#odus.nothing.get_universal_neutral_for_methods)(method_names) | See get_universal_neutral_for_method(method_name).                               |
| `transparent_method`(self, x)                                                                    |                                                                                  |

### Classes

| [`GreatUniversalNothing`](#odus.nothing.GreatUniversalNothing)()   | An object that is a neutral for any method.   |
|----------------------------------------------------------------------------|-----------------------------------------------|

### *class* odus.nothing.GreatUniversalNothing

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An object that is a neutral for any method.
Only works with some operators (e.g. works for \_\_add_\_ works, but not with +).

```pycon
>>> nothing = GreatUniversalNothing()  # note, this is constructed ones, and used many times
>>>
>>> nothing + 2
2
>>> 2 + nothing
2
>>> nothing + 2 + nothing
2
>>> nothing * 3.14 * nothing
3.14
>>> nothing | [1, 2, 3] | nothing
[1, 2, 3]
>>> nothing & 'hi there' & nothing
'hi there'
>>> # Now we demo how we get the same behavior no matter what the type is
>>> nothing + 2
2
>>> nothing + [1, 2, 3]
[1, 2, 3]
>>> print((nothing + "wassup world!"))
wassup world!
>>>
>>> # Useful as an alternative of using zeros(shape_of_your_array) to initialized array cumulators.
>>> # No need to specify the array shape in advance!
>>> import numpy as np
>>> print((nothing + np.array([1, 2, 3, 4])))
[1 2 3 4]
>>> # and += works automatically!
>>> t = nothing
>>> t += np.array([1, 2, 3, 4])
>>> print(t)
[1 2 3 4]
>>> t += np.array([10, 20, 30, 40])
>>> print(t)
[11 22 33 44]
>>>
>>> # Example with defaultdict (and multi type value cumulators
>>> from collections import defaultdict
>>> t = defaultdict(lambda: nothing)
>>> t['foo'] += 2
>>> t['bar'] += [1,2,3]
>>> t['bar'] += [10,20]
>>> t['is'] += np.array([1,2,3,4])
>>> t['is'] += np.array([1,2,3,4])
>>> t['overused'] += "hello"
>>> t['overused'] += " "
>>> t['overused'] += "world"
>>> dict(t)
{'foo': 2, 'bar': [1, 2, 3, 10, 20], 'is': array([2, 4, 6, 8]), 'overused': 'hello world'}
>>>
>>> # Works with other methods too!
>>> nothing * 314
314
>>> nothing * 'foobar'
'foobar'
```

### odus.nothing.get_universal_neutral_for_method(method_name)

A universal neutral element (synonyms: “identity element” and “unit element”).
A neutral element is with respect to a binary operation op(a, b).
A neutral element e (for op) is one such that op(e, b) = b for any b.
(Formally, we should also have op(b, e) = b, but we’ll ignore that!)

It is universal from the point of view of type, not method.
Here, you specify what method (operation) you want it to be a neutral element for.
Typically, you use this when you want to initialize a variable that will “accumulate/aggregate” objects one by one.
For example, you say

> cumul = 0

and then add elements to cumul
: cumul = cumul + x

or
: cumul += x

etc.

Depending on what the type is, you need to initialize in different ways.
: cumul = 0   for ints
  cumul = []  for lists
  cumul = numpy.array(array_shape) for numpy arrays (and better know the dims of the arrays you’ll be feeding!)
  cumul = “” for strings

With the universal element, you don’t need to initialize according to type. You just need to say:

```default
cumul = get_universal_neutral_for_method("__add__")
```

and you’ll have something that behaves as expected: Be transparent.

Other examples of common methods where you want a neutral:  \_\_mul_\_, update, etc.

```pycon
>>> nothing = get_universal_neutral_for_method('__add__')  # note, this is constructed ones, and used many times
>>>
>>> nothing + 2
2
>>> nothing + [1, 2, 3]
[1, 2, 3]
>>> print((nothing + "wassup world!"))
wassup world!
>>>
>>> # Useful as an alternative of using zeros(shape_of_your_array) to initialized array cumulators.
>>> # No need to specify the array shape in advance!
>>> import numpy as np
>>> print((nothing + np.array([1, 2, 3, 4])))
[1 2 3 4]
>>> # and += works automatically!
>>> t = nothing
>>> t += np.array([1, 2, 3, 4])
>>> print(t)
[1 2 3 4]
>>> t += np.array([10, 20, 30, 40])
>>> print(t)
[11 22 33 44]
>>>
>>> # Example with defaultdict (and multi type value cumulators
>>> from collections import defaultdict
>>> t = defaultdict(lambda: nothing)
>>> t['foo'] += 2
>>> t['bar'] += [1,2,3]
>>> t['bar'] += [10,20]
>>> t['is'] += np.array([1,2,3,4])
>>> t['is'] += np.array([1,2,3,4])
>>> t['overused'] += "hello"
>>> t['overused'] += " "
>>> t['overused'] += "world"
>>> dict(t)
{'foo': 2, 'bar': [1, 2, 3, 10, 20], 'is': array([2, 4, 6, 8]), 'overused': 'hello world'}
>>>
>>> # Works with other methods too!
>>> nothing = get_universal_neutral_for_method('__mul__')
>>> nothing * 314
314
>>> nothing * 'foobar'
'foobar'
```

### odus.nothing.get_universal_neutral_for_methods(method_names)

See get_universal_neutral_for_method(method_name).
This is just the “multiple methods” version of that one.
