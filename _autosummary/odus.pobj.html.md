# odus.pobj

Dynamic method injection on objects and classes.

`add_method` returns a new object whose class has an extra method, and
`inject_method` adds methods to an existing object in place.

### Functions

| [`add_method`](#odus.pobj.add_method)(obj, method_func[, method_name, ...])   | Dynamically add a method to an object.      |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------|
| [`inject_method`](#odus.pobj.inject_method)(obj, method_func[, method_name])     | Inject one or several methods to an object. |

### odus.pobj.add_method(obj, method_func, method_name=None, class_name=None)

Dynamically add a method to an object.

* **Parameters:**
  * **obj** – The object to add a method to
  * **method_func** – The function to use as a method. The first argument must be the object itself
    (usually called self)
  * **method_name** – The desired function name. If None, will take method_func._\_name_\_
  * **class_name** – The desired class name. If None, will take type(obj)._\_name_\_
* **Returns:**
  the object, but with the additional method (or a different function for it)

```pycon
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
```

### odus.pobj.inject_method(obj, method_func, method_name=None)

Inject one or several methods to an object.
Note that it doesn’t work if method_name is a dunder.
If you want to inject things like \_\_call_\_, \_\_getitem_\_, etc., use add_method instead.

* **Parameters:**
  * **obj** – The object to add a method to
  * **method_func** – The function to use as a method. The first argument must be the object itself
    (usually called self)
  * **method_name** – The desired function name. If None, will take method_func._\_name_\_
* **Returns:**
  the object, but with the additional method (or a different function for it)

```pycon
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
```
