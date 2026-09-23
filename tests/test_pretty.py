import collections
import io
import sys
from array import array
from collections import UserDict, defaultdict, deque
from dataclasses import InitVar, dataclass, field
from typing import Any, ClassVar, List, NamedTuple

import attr
import pytest

from rich.console import Console
from rich.measure import Measurement
from rich.pretty import (
    Node,
    Pretty,
    _ipy_display_hook,
    install,
    is_expandable,
    pprint,
    pretty_repr,
)
from rich.text import Text

skip_py38 = pytest.mark.skipif(
    sys.version_info.minor == 8 and sys.version_info.major == 3,
    reason="rendered differently on py3.8",
)
skip_py39 = pytest.mark.skipif(
    sys.version_info.minor == 9 and sys.version_info.major == 3,
    reason="rendered differently on py3.9",
)
skip_py310 = pytest.mark.skipif(
    sys.version_info.minor == 10 and sys.version_info.major == 3,
    reason="rendered differently on py3.10",
)
skip_py311 = pytest.mark.skipif(
    sys.version_info.minor == 11 and sys.version_info.major == 3,
    reason="rendered differently on py3.11",
)
skip_py312 = pytest.mark.skipif(
    sys.version_info.minor == 12 and sys.version_info.major == 3,
    reason="rendered differently on py3.12",
)
skip_py313 = pytest.mark.skipif(
    sys.version_info.minor == 13 and sys.version_info.major == 3,
    reason="rendered differently on py3.13",
)
skip_py314 = pytest.mark.skipif(
    sys.version_info.minor == 14 and sys.version_info.major == 3,
    reason="rendered differently on py3.14",
)


def test_install() -> None:
    console = Console(file=io.StringIO())
    dh = sys.displayhook
    install(console)
    sys.displayhook("foo")
    assert console.file.getvalue() == "'foo'\n"
    assert sys.displayhook is not dh


def test_install_max_depth() -> None:
    console = Console(file=io.StringIO())
    dh = sys.displayhook
    install(console, max_depth=1)
    sys.displayhook({"foo": {"bar": True}})
    assert console.file.getvalue() == "{'foo': {...}}\n"
    assert sys.displayhook is not dh


def test_ipy_display_hook__repr_html() -> None:
    console = Console(file=io.StringIO(), force_jupyter=True)

    class Thing:
        def _repr_html_(self):
            return "hello"

    console.begin_capture()
    _ipy_display_hook(Thing(), console=console)

    # Rendering delegated to notebook because _repr_html_ method exists
    assert console.end_capture() == ""


def test_ipy_display_hook__multiple_special_reprs() -> None:
    """
    The case where there are multiple IPython special _repr_*_
    methods on the object, and one of them returns None but another
    one does not.
    """
    console = Console(file=io.StringIO(), force_jupyter=True)

    class Thing:
        def __repr__(self):
            return "A Thing"

        def _repr_latex_(self):
            return None

        def _repr_html_(self):
            return "hello"

    result = _ipy_display_hook(Thing(), console=console)
    assert result == "A Thing"


def test_ipy_display_hook__no_special_repr_methods() -> None:
    console = Console(file=io.StringIO(), force_jupyter=True)

    class Thing:
        def __repr__(self) -> str:
            return "hello"

    result = _ipy_display_hook(Thing(), console=console)
    # should be repr as-is
    assert result == "hello"


def test_ipy_display_hook__special_repr_raises_exception() -> None:
    """
    When an IPython special repr method raises an exception,
    we treat it as if it doesn't exist and look for the next.
    """
    console = Console(file=io.StringIO(), force_jupyter=True)

    class Thing:
        def _repr_markdown_(self):
            raise Exception()

        def _repr_latex_(self):
            return None

        def _repr_html_(self):
            return "hello"

        def __repr__(self):
            return "therepr"

    result = _ipy_display_hook(Thing(), console=console)
    assert result == "therepr"


def test_ipy_display_hook__console_renderables_on_newline() -> None:
    console = Console(file=io.StringIO(), force_jupyter=True)
    console.begin_capture()
    result = _ipy_display_hook(Text("hello"), console=console)
    assert result == "\nhello"


def test_pretty() -> None:
    test = {
        "foo": [1, 2, 3, (4, 5, {6}, 7, 8, {9}), {}],
        "bar": {"egg": "baz", "words": ["Hello World"] * 10},
        False: "foo",
        True: "",
        "text": ("Hello World", "foo bar baz egg"),
    }

    result = pretty_repr(test, max_width=80)
    print(result)
    expected = "{\n    'foo': [1, 2, 3, (4, 5, {6}, 7, 8, {9}), {}],\n    'bar': {\n        'egg': 'baz',\n        'words': [\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World',\n            'Hello World'\n        ]\n    },\n    False: 'foo',\n    True: '',\n    'text': ('Hello World', 'foo bar baz egg')\n}"
    print(expected)
    assert result == expected


@dataclass
class ExampleDataclass:
    foo: int
    bar: str
    ignore: int = field(repr=False)
    baz: List[str] = field(default_factory=list)
    last: int = field(default=1, repr=False)


@dataclass
class Empty:
    pass


def test_pretty_dataclass() -> None:
    dc = ExampleDataclass(1000, "Hello, World", 999, ["foo", "bar", "baz"])
    result = pretty_repr(dc, max_width=80)
    print(repr(result))
    assert (
        result
        == "ExampleDataclass(foo=1000, bar='Hello, World', baz=['foo', 'bar', 'baz'])"
    )
    result = pretty_repr(dc, max_width=16)
    print(repr(result))
    assert (
        result
        == "ExampleDataclass(\n    foo=1000,\n    bar='Hello, World',\n    baz=[\n        'foo',\n        'bar',\n        'baz'\n    ]\n)"
    )
    dc.bar = dc
    result = pretty_repr(dc, max_width=80)
    print(repr(result))
    assert result == "ExampleDataclass(foo=1000, bar=..., baz=['foo', 'bar', 'baz'])"


def test_empty_dataclass() -> None:
    assert pretty_repr(Empty()) == "Empty()"
    assert pretty_repr([Empty()]) == "[Empty()]"


class StockKeepingUnit(NamedTuple):
    name: str
    description: str
    price: float
    category: str
    reviews: List[str]


def test_pretty_namedtuple() -> None:
    console = Console(color_system=None)
    console.begin_capture()

    example_namedtuple = StockKeepingUnit(
        "Sparkling British Spring Water",
        "Carbonated spring water",
        0.9,
        "water",
        ["its amazing!", "its terrible!"],
    )

    result = pretty_repr(example_namedtuple)

    print(result)
    assert (
        result
        == """StockKeepingUnit(
    name='Sparkling British Spring Water',
    description='Carbonated spring water',
    price=0.9,
    category='water',
    reviews=['its amazing!', 'its terrible!']
)"""
    )


def test_pretty_namedtuple_length_one_no_trailing_comma() -> None:
    instance = collections.namedtuple("Thing", ["name"])(name="Bob")
    assert pretty_repr(instance) == "Thing(name='Bob')"


def test_pretty_namedtuple_empty() -> None:
    instance = collections.namedtuple("Thing", [])()
    assert pretty_repr(instance) == "Thing()"


def test_pretty_namedtuple_custom_repr() -> None:
    class Thing(NamedTuple):
        def __repr__(self):
            return "XX"

    assert pretty_repr(Thing()) == "XX"


def test_pretty_namedtuple_fields_invalid_type() -> None:
    class LooksLikeANamedTupleButIsnt(tuple):
        _fields = "blah"

    instance = LooksLikeANamedTupleButIsnt()
    result = pretty_repr(instance)
    assert result == "()"  # Treated as tuple


def test_pretty_repr_does_not_mutate_auto_vivifying_object() -> None:
    """Feature detection must not leave the probe attributes it creates behind."""

    class AutoVivifying:
        def __getattr__(self, name: str) -> "AutoVivifying":
            value = AutoVivifying()
            setattr(self, name, value)
            return value

        def __repr__(self) -> str:
            return "AutoVivifying()"

    instance = AutoVivifying()
    assert vars(instance) == {}

    assert pretty_repr(instance) == "AutoVivifying()"
    assert vars(instance) == {}


def test_pretty_repr_does_not_mutate_auto_vivifying_mapping() -> None:
    """A dict-like object must not acquire Rich's probe names as keys."""

    class AutoVivifyingDict(dict):
        def __getattr__(self, name: str) -> "AutoVivifyingDict":
            value = AutoVivifyingDict()
            self[name] = value
            return value

        def __delattr__(self, name: str) -> None:
            del self[name]

    instance = AutoVivifyingDict()
    assert pretty_repr(instance) == "{}"
    assert instance == {}


def test_pretty_repr_does_not_execute_failing_getattr() -> None:
    """Static feature checks must not call an unrelated failing hook."""

    accessed = []

    class FailingGetattr:
        def __getattr__(self, name: str) -> Any:
            accessed.append(name)
            raise RuntimeError(name)

        def __repr__(self) -> str:
            return "FailingGetattr()"

    assert pretty_repr(FailingGetattr()) == "FailingGetattr()"
    assert accessed == []


def test_pretty_repr_does_not_execute_failing_getattribute() -> None:
    accessed = []

    class FailingGetattribute:
        def __getattribute__(self, name: str) -> Any:
            if name != "__repr__":
                accessed.append(name)
                raise RuntimeError(name)
            return object.__getattribute__(self, name)

        def __repr__(self) -> str:
            return "FailingGetattribute()"

    assert pretty_repr(FailingGetattribute()) == "FailingGetattribute()"
    assert accessed == []


def test_pretty_repr_supports_slots_without_dynamic_access() -> None:
    """Slots are real fields, but looking for Rich protocols must stay static."""

    @dataclass
    class Slotted:
        __slots__ = ("value",)

        value: int

        def __getattr__(self, name: str) -> Any:
            raise AssertionError("unexpected attribute access: " + name)

    instance = Slotted(1)
    assert pretty_repr(instance) == "Slotted(value=1)"


def test_namedtuple_probe_does_not_execute_dynamic_fields() -> None:
    """A tuple subclass with a dynamic _fields must remain an ordinary tuple."""

    class DynamicFieldsTuple(tuple):
        def __getattr__(self, name: str) -> Any:
            setattr(self, name, ("invented",))
            return getattr(self, name)

    instance = DynamicFieldsTuple((1, 2))
    assert pretty_repr(instance) == "(1, 2)"
    assert vars(instance) == {}


def test_namedtuple_probe_does_not_execute_fields_descriptor() -> None:
    """A descriptor named _fields is not evidence of a namedtuple."""

    accessed = []

    class FieldsDescriptor:
        def __get__(self, instance: Any, owner: Any) -> Any:
            accessed.append(owner)
            raise AssertionError("_fields descriptor was executed")

    class DescriptorTuple(tuple):
        _fields = FieldsDescriptor()

    assert pretty_repr(DescriptorTuple((1, 2))) == "(1, 2)"
    assert accessed == []


def test_namedtuple_probe_ignores_metaclass_fields() -> None:
    """Metaclass attributes must not turn an ordinary tuple into a namedtuple."""

    class Meta(type):
        _fields = ("invented",)

    class MetaclassTuple(tuple, metaclass=Meta):
        pass

    assert pretty_repr(MetaclassTuple((1,))) == "(1)"


def test_namedtuple_probe_requires_matching_field_count() -> None:
    """Malformed class-level metadata must not discard tuple values."""

    class MalformedTuple(tuple):
        _fields = ("first",)

    assert pretty_repr(MalformedTuple((1, 2))) == "(1, 2)"


def test_namedtuple_rendering_does_not_execute_overridden_asdict() -> None:
    """Tuple values are available without trusting an overridden _asdict hook."""

    BaseNamedTuple = collections.namedtuple("BaseNamedTuple", ["value"])
    accessed = []

    class HostileNamedTuple(BaseNamedTuple):
        def _asdict(self) -> Any:
            accessed.append(True)
            raise AssertionError("_asdict hook was executed")

    instance = HostileNamedTuple(1)
    assert pretty_repr(instance) == "HostileNamedTuple(value=1)"
    assert accessed == []


def test_declared_rich_repr_descriptor_is_bound_only_when_rendered() -> None:
    """Static discovery does not bind an explicitly declared Rich repr."""

    accessed = []

    class RichReprDescriptor:
        def __get__(self, instance: Any, owner: Any) -> Any:
            accessed.append("bind")

            def rich_repr():
                accessed.append("call")
                yield "value", 1

            return rich_repr

    class DeclaredRepresentation:
        __rich_repr__ = RichReprDescriptor()

        def __repr__(self) -> str:
            return "DeclaredRepresentation()"

    instance = DeclaredRepresentation()
    assert is_expandable(instance)
    assert accessed == []
    assert pretty_repr(instance) == "DeclaredRepresentation(value=1)"
    assert accessed == ["bind", "call"]


def test_failing_rich_repr_iterator_falls_back_to_repr() -> None:
    accessed = []

    class BrokenRepresentation:
        def __rich_repr__(self):
            accessed.append("call")

            def values():
                accessed.append("iterate")
                raise RuntimeError("iteration failed")
                yield  # pragma: no cover

            return values()

        def __repr__(self) -> str:
            return "BrokenRepresentation()"

    assert pretty_repr(BrokenRepresentation()) == "BrokenRepresentation()"
    assert accessed == ["call", "iterate"]


def test_dynamic_rich_repr_is_not_probed() -> None:
    """Dynamic protocol synthesis must not run during pretty introspection."""

    accessed = []

    class DynamicRepresentation:
        def __getattr__(self, name: str) -> Any:
            accessed.append(name)
            if name == "__rich_repr__":
                return lambda: [("value", 1)]
            raise AttributeError(name)

        def __repr__(self) -> str:
            return "DynamicRepresentation()"

    instance = DynamicRepresentation()
    assert pretty_repr(instance) == "DynamicRepresentation()"
    assert accessed == []


def test_genuine_namedtuple_variants_remain_expandable() -> None:
    """Static _fields detection must preserve both supported namedtuple forms."""

    CollectionPoint = collections.namedtuple("CollectionPoint", ["x"])

    class TypedPoint(NamedTuple):
        y: int

    value = {"collection": CollectionPoint(1), "typed": TypedPoint(2)}
    assert pretty_repr(value, max_width=200) == (
        "{'collection': CollectionPoint(x=1), 'typed': TypedPoint(y=2)}"
    )


def test_nested_namedtuples_cycles_width_and_depth() -> None:
    """Nested supported objects retain cycle, width, and depth behavior."""

    CollectionNode = collections.namedtuple("CollectionNode", ["child"])

    class TypedNode(NamedTuple):
        child: Any

    root = {}
    root["collection"] = CollectionNode(root)
    root["typed"] = TypedNode(root)

    narrow = pretty_repr(root, max_width=30)
    assert "CollectionNode(" in narrow
    assert "TypedNode(" in narrow
    assert narrow.count("child=...") == 2
    assert "\n" in narrow

    assert pretty_repr(root, max_width=200, max_depth=1) == (
        "{'collection': CollectionNode(...), 'typed': TypedNode(...)}"
    )


def test_custom_rich_repr_is_still_used_for_nested_values() -> None:
    accessed = []

    class CustomRepresentation:
        def __getattr__(self, name: str) -> Any:
            accessed.append(name)
            raise AssertionError(name)

        def __rich_repr__(self):
            yield "values", {"nested": [1, 2]}

    assert pretty_repr(CustomRepresentation(), max_width=200) == (
        "CustomRepresentation(values={'nested': [1, 2]})"
    )
    assert accessed == []


def test_instance_rich_repr_remains_supported() -> None:
    class InstanceRepresentation:
        __slots__ = ("__rich_repr__",)

        def __init__(self) -> None:
            self.__rich_repr__ = lambda: [("value", 8)]

    assert pretty_repr(InstanceRepresentation()) == "InstanceRepresentation(value=8)"


@pytest.mark.parametrize(
    "decorate", [dataclass, attr.define(slots=False)], ids=["dataclass", "attrs"]
)
@pytest.mark.parametrize("fail", [False, True], ids=["value", "error"])
def test_declared_field_getattr_is_read_once(decorate: Any, fail: bool) -> None:
    accessed = []

    @decorate
    class Example:
        value: int

        def __getattr__(self, name: str) -> int:
            accessed.append(name)
            if name != "value":
                raise AttributeError(name)
            if fail:
                raise RuntimeError("field failed")
            return 7

    instance = Example(1)
    del instance.value
    expected = "RuntimeError('field failed')" if fail else "7"
    assert pretty_repr(instance) == f"Example(value={expected})"
    assert accessed == ["value"]


@pytest.mark.parametrize(
    "decorate", [dataclass, attr.define(slots=False)], ids=["dataclass", "attrs"]
)
@pytest.mark.parametrize("fail", [False, True], ids=["value", "error"])
def test_declared_field_descriptor_is_read_once(decorate: Any, fail: bool) -> None:
    accessed = []

    @decorate
    class Example:
        value: int

    def read_value(instance: Any) -> int:
        accessed.append("value")
        if fail:
            raise RuntimeError("field failed")
        return 7

    instance = Example(1)
    # A data descriptor takes precedence over the stored field value.
    Example.value = property(read_value)
    expected = "RuntimeError('field failed')" if fail else "7"
    assert pretty_repr(instance) == f"Example(value={expected})"
    assert accessed == ["value"]


@pytest.mark.parametrize("deleted", [False, True], ids=["missing", "deleted"])
def test_dict_backed_attrs_missing_field_displays_error(deleted: bool) -> None:
    @attr.define(slots=False)
    class Example:
        value: int = attr.field(init=False)

    instance = Example()
    if deleted:
        instance.value = 1
        del instance.value

    # AttributeError's message includes the qualified class name on Python 3.13+.
    with pytest.raises(AttributeError) as error:
        instance.value
    assert pretty_repr(instance, max_width=200) == f"Example(value={error.value!r})"


def test_dataclass_shadowed_dict_preserves_fields() -> None:
    @dataclass
    class Example:
        __dict__: dict
        value: int = 1

    assert pretty_repr(Example({})) == "Example(__dict__={'value': 1}, value=1)"


def test_dataclass_filters_fields_before_reading() -> None:
    accessed = []

    @dataclass
    class Example:
        value: int
        init_only: InitVar[int] = 2
        class_only: ClassVar[int] = 3
        hidden: int = field(default=4, repr=False)

        def __getattribute__(self, name: str) -> Any:
            accessed.append(name)
            return object.__getattribute__(self, name)

    assert pretty_repr(Example(1)) == "Example(value=1)"
    assert accessed == ["value"]


def test_dict_backed_instance_rich_repr_remains_supported() -> None:
    accessed = []

    class InstanceRepresentation:
        def __getattr__(self, name: str) -> Any:
            accessed.append(name)
            raise AttributeError(name)

    def rich_repr():
        accessed.append("call")
        yield "value", 8

    instance = InstanceRepresentation()
    instance.__rich_repr__ = rich_repr
    assert vars(instance)["__rich_repr__"] is rich_repr
    assert is_expandable(instance)
    assert accessed == []
    assert pretty_repr(instance) == "InstanceRepresentation(value=8)"
    assert accessed == ["call"]


def test_pretty_namedtuple_max_depth() -> None:
    instance = {"unit": StockKeepingUnit("a", "b", 1.0, "c", ["d", "e"])}
    result = pretty_repr(instance, max_depth=1)
    assert result == "{'unit': StockKeepingUnit(...)}"


def test_small_width() -> None:
    test = ["Hello world! 12345"]
    result = pretty_repr(test, max_width=10)
    expected = "[\n    'Hello world! 12345'\n]"
    assert result == expected


def test_ansi_in_pretty_repr() -> None:
    class Hello:
        def __repr__(self):
            return "Hello \x1b[38;5;239mWorld!"

    pretty = Pretty(Hello())

    console = Console(file=io.StringIO(), record=True)
    console.print(pretty)
    result = console.export_text()

    assert result == "Hello World!\n"


def test_broken_repr() -> None:
    class BrokenRepr:
        def __repr__(self):
            1 / 0

    test = [BrokenRepr()]
    result = pretty_repr(test)
    expected = "[<repr-error 'division by zero'>]"
    assert result == expected


def test_broken_getattr() -> None:
    class BrokenAttr:
        def __getattr__(self, name):
            1 / 0

        def __repr__(self):
            return "BrokenAttr()"

    test = BrokenAttr()
    result = pretty_repr(test)
    assert result == "BrokenAttr()"


def test_reference_cycle_container() -> None:
    test = []
    test.append(test)
    res = pretty_repr(test)
    assert res == "[...]"

    test = [1, []]
    test[1].append(test)
    res = pretty_repr(test)
    assert res == "[1, [...]]"

    # Not a cyclic reference, just a repeated reference
    a = [2]
    test = [1, [a, a]]
    res = pretty_repr(test)
    assert res == "[1, [[2], [2]]]"


def test_reference_cycle_namedtuple() -> None:
    class Example(NamedTuple):
        x: int
        y: Any

    test = Example(1, [Example(2, [])])
    test.y[0].y.append(test)
    res = pretty_repr(test)
    assert res == "Example(x=1, y=[Example(x=2, y=[...])])"

    # Not a cyclic reference, just a repeated reference
    a = Example(2, None)
    test = Example(1, [a, a])
    res = pretty_repr(test)
    assert res == "Example(x=1, y=[Example(x=2, y=None), Example(x=2, y=None)])"


def test_reference_cycle_dataclass() -> None:
    @dataclass
    class Example:
        x: int
        y: Any

    test = Example(1, None)
    test.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=...)"

    test = Example(1, Example(2, None))
    test.y.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=Example(x=2, y=...))"

    # Not a cyclic reference, just a repeated reference
    a = Example(2, None)
    test = Example(1, [a, a])
    res = pretty_repr(test)
    assert res == "Example(x=1, y=[Example(x=2, y=None), Example(x=2, y=None)])"


def test_reference_cycle_attrs() -> None:
    @attr.define
    class Example:
        x: int
        y: Any

    test = Example(1, None)
    test.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=...)"

    test = Example(1, Example(2, None))
    test.y.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=Example(x=2, y=...))"

    # Not a cyclic reference, just a repeated reference
    a = Example(2, None)
    test = Example(1, [a, a])
    res = pretty_repr(test)
    assert res == "Example(x=1, y=[Example(x=2, y=None), Example(x=2, y=None)])"


def test_reference_cycle_custom_repr() -> None:
    class Example:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __rich_repr__(self):
            yield ("x", self.x)
            yield ("y", self.y)

    test = Example(1, None)
    test.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=...)"

    test = Example(1, Example(2, None))
    test.y.y = test
    res = pretty_repr(test)
    assert res == "Example(x=1, y=Example(x=2, y=...))"

    # Not a cyclic reference, just a repeated reference
    a = Example(2, None)
    test = Example(1, [a, a])
    res = pretty_repr(test)
    assert res == "Example(x=1, y=[Example(x=2, y=None), Example(x=2, y=None)])"


def test_max_depth() -> None:
    d = {}
    d["foo"] = {"fob": {"a": [1, 2, 3], "b": {"z": "x", "y": ["a", "b", "c"]}}}

    assert pretty_repr(d, max_depth=0) == "{...}"
    assert pretty_repr(d, max_depth=1) == "{'foo': {...}}"
    assert pretty_repr(d, max_depth=2) == "{'foo': {'fob': {...}}}"
    assert pretty_repr(d, max_depth=3) == "{'foo': {'fob': {'a': [...], 'b': {...}}}}"
    assert (
        pretty_repr(d, max_width=100, max_depth=4)
        == "{'foo': {'fob': {'a': [1, 2, 3], 'b': {'z': 'x', 'y': [...]}}}}"
    )
    assert (
        pretty_repr(d, max_width=100, max_depth=5)
        == "{'foo': {'fob': {'a': [1, 2, 3], 'b': {'z': 'x', 'y': ['a', 'b', 'c']}}}}"
    )
    assert (
        pretty_repr(d, max_width=100, max_depth=None)
        == "{'foo': {'fob': {'a': [1, 2, 3], 'b': {'z': 'x', 'y': ['a', 'b', 'c']}}}}"
    )


def test_max_depth_rich_repr() -> None:
    class Foo:
        def __init__(self, foo):
            self.foo = foo

        def __rich_repr__(self):
            yield "foo", self.foo

    class Bar:
        def __init__(self, bar):
            self.bar = bar

        def __rich_repr__(self):
            yield "bar", self.bar

    assert (
        pretty_repr(Foo(foo=Bar(bar=Foo(foo=[]))), max_depth=2)
        == "Foo(foo=Bar(bar=Foo(...)))"
    )


def test_max_depth_attrs() -> None:
    @attr.define
    class Foo:
        foo = attr.field()

    @attr.define
    class Bar:
        bar = attr.field()

    assert (
        pretty_repr(Foo(foo=Bar(bar=Foo(foo=[]))), max_depth=2)
        == "Foo(foo=Bar(bar=Foo(...)))"
    )


def test_max_depth_dataclass() -> None:
    @dataclass
    class Foo:
        foo: object

    @dataclass
    class Bar:
        bar: object

    assert (
        pretty_repr(Foo(foo=Bar(bar=Foo(foo=[]))), max_depth=2)
        == "Foo(foo=Bar(bar=Foo(...)))"
    )


def test_defaultdict() -> None:
    test_dict = defaultdict(int, {"foo": 2})
    result = pretty_repr(test_dict)
    assert result == "defaultdict(<class 'int'>, {'foo': 2})"


def test_deque() -> None:
    test_deque = deque([1, 2, 3])
    result = pretty_repr(test_deque)
    assert result == "deque([1, 2, 3])"
    test_deque = deque([1, 2, 3], maxlen=None)
    result = pretty_repr(test_deque)
    assert result == "deque([1, 2, 3])"
    test_deque = deque([1, 2, 3], maxlen=5)
    result = pretty_repr(test_deque)
    assert result == "deque([1, 2, 3], maxlen=5)"
    test_deque = deque([1, 2, 3], maxlen=0)
    result = pretty_repr(test_deque)
    assert result == "deque(maxlen=0)"
    test_deque = deque([])
    result = pretty_repr(test_deque)
    assert result == "deque()"
    test_deque = deque([], maxlen=None)
    result = pretty_repr(test_deque)
    assert result == "deque()"
    test_deque = deque([], maxlen=5)
    result = pretty_repr(test_deque)
    assert result == "deque(maxlen=5)"
    test_deque = deque([], maxlen=0)
    result = pretty_repr(test_deque)
    assert result == "deque(maxlen=0)"


def test_array() -> None:
    test_array = array("I", [1, 2, 3])
    result = pretty_repr(test_array)
    assert result == "array('I', [1, 2, 3])"


def test_tuple_of_one() -> None:
    assert pretty_repr((1,)) == "(1,)"


def test_node() -> None:
    node = Node("abc")
    assert pretty_repr(node) == "abc: "


def test_indent_lines() -> None:
    console = Console(width=100, color_system=None)
    console.begin_capture()
    console.print(Pretty([100, 200], indent_guides=True), width=8)
    expected = """\
[
│   100,
│   200
]
"""
    result = console.end_capture()
    print(repr(result))
    print(result)
    assert result == expected


def test_pprint() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    pprint(1, console=console)
    assert console.end_capture() == "1\n"


def test_pprint_max_values() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    pprint([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], console=console, max_length=2)
    assert console.end_capture() == "[1, 2, ... +8]\n"


def test_pprint_max_items() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    pprint({"foo": 1, "bar": 2, "egg": 3}, console=console, max_length=2)
    assert console.end_capture() == """{'foo': 1, 'bar': 2, ... +1}\n"""


def test_pprint_max_string() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    pprint(["Hello" * 20], console=console, max_string=8)
    assert console.end_capture() == """['HelloHel'+92]\n"""


def test_tuples() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    pprint((1,), console=console)
    pprint((1,), expand_all=True, console=console)
    pprint(((1,),), expand_all=True, console=console)
    result = console.end_capture()
    print(repr(result))
    expected = "(1,)\n(\n│   1,\n)\n(\n│   (\n│   │   1,\n│   ),\n)\n"
    print(result)
    print("--")
    print(expected)
    assert result == expected


def test_newline() -> None:
    console = Console(color_system=None)
    console.begin_capture()
    console.print(Pretty((1,), insert_line=True, expand_all=True))
    result = console.end_capture()
    expected = "\n(\n    1,\n)\n"
    assert result == expected


def test_empty_repr() -> None:
    class Foo:
        def __repr__(self):
            return ""

    assert pretty_repr(Foo()) == ""


def test_attrs() -> None:
    @attr.define
    class Point:
        x: int
        y: int
        foo: str = attr.field(repr=str.upper)
        z: int = 0

    result = pretty_repr(Point(1, 2, foo="bar"))
    print(repr(result))
    expected = "Point(x=1, y=2, foo=BAR, z=0)"
    assert result == expected


def test_attrs_empty() -> None:
    @attr.define
    class Nada:
        pass

    result = pretty_repr(Nada())
    print(repr(result))
    expected = "Nada()"
    assert result == expected


@skip_py310
@skip_py311
@skip_py312
@skip_py313
@skip_py314
def test_attrs_broken() -> None:
    @attr.define
    class Foo:
        bar: int

    foo = Foo(1)
    del foo.bar
    result = pretty_repr(foo)
    print(repr(result))
    expected = "Foo(bar=AttributeError('bar'))"
    assert result == expected


@skip_py38
@skip_py39
def test_attrs_broken_310() -> None:
    @attr.define
    class Foo:
        bar: int

    foo = Foo(1)
    del foo.bar
    result = pretty_repr(foo)
    print(repr(result))
    if sys.version_info >= (3, 13):
        expected = "Foo(\n    bar=AttributeError(\"'tests.test_pretty.test_attrs_broken_310.<locals>.Foo' object has no attribute 'bar'\")\n)"
    else:
        expected = "Foo(bar=AttributeError(\"'Foo' object has no attribute 'bar'\"))"
    assert result == expected


def test_user_dict() -> None:
    class D1(UserDict):
        pass

    class D2(UserDict):
        def __repr__(self):
            return "FOO"

    d1 = D1({"foo": "bar"})
    d2 = D2({"foo": "bar"})
    result = pretty_repr(d1, expand_all=True)
    print(repr(result))
    assert result == "{\n    'foo': 'bar'\n}"
    result = pretty_repr(d2, expand_all=True)
    print(repr(result))
    assert result == "FOO"


def test_lying_attribute() -> None:
    """Test getattr doesn't break rich repr protocol"""

    class Foo:
        def __getattr__(self, attr):
            return "foo"

    foo = Foo()
    result = pretty_repr(foo)
    assert "Foo" in result


def test_measure_pretty() -> None:
    """Test measure respects expand_all"""
    # https://github.com/Textualize/rich/issues/1998
    console = Console()
    pretty = Pretty(["alpha", "beta", "delta", "gamma"], expand_all=True)

    measurement = console.measure(pretty)
    assert measurement == Measurement(12, 12)


def test_tuple_rich_repr() -> None:
    """
    Test that can use None as key to have tuple positional values.
    """

    class Foo:
        def __rich_repr__(self):
            yield None, (1,)

    assert pretty_repr(Foo()) == "Foo((1,))"


def test_tuple_rich_repr_default() -> None:
    """
    Test that can use None as key to have tuple positional values and with a default.
    """

    class Foo:
        def __rich_repr__(self):
            yield None, (1,), (1,)

    assert pretty_repr(Foo()) == "Foo()"


def test_dataclass_no_attribute() -> None:
    """Regression test for https://github.com/Textualize/rich/issues/3417"""
    from dataclasses import dataclass, field

    @dataclass(eq=False)
    class BadDataclass:
        item: int = field(init=False)

    # item is not provided
    bad_data_class = BadDataclass()

    console = Console()
    with console.capture() as capture:
        console.print(bad_data_class)

    expected = "BadDataclass()\n"
    result = capture.get()
    assert result == expected
