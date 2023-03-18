import operator
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union, Optional

from .sample import Sample


def _unpack_nested_value(container: Union[Sample, Dict[str, Any]], attr_path: List[str]) -> Any:
    if isinstance(container, dict):
        if attr_path[0] not in container:
            raise KeyError(f"Attempted to access [{attr_path[0]}], but it does not exist.")
        value = container[attr_path[0]]
    elif isinstance(container, Sample):
        if not hasattr(container, attr_path[0]):
            raise KeyError(f"Attempted to access [{attr_path[0]}], but it does not exist.")
        value = getattr(container, attr_path[0])
    else:
        raise TypeError("Can only unpack dict or Sample objects.")

    if isinstance(value, dict) and len(attr_path) > 1:
        return _unpack_nested_value(value, attr_path[1:])
    else:
        return value


@dataclass
class FilterBase:

    attr_path: List[str]

    def __call__(self, element: Sample) -> bool:
        raise NotImplementedError


@dataclass
class InclusionExclusionFilter(FilterBase):

    attr_path: List[str]
    include: List[Any] = field(default_factory=list)
    exclude: List[Any] = field(default_factory=list)

    def __call__(self, element: Union[Sample, Dict[str, Any]]) -> bool:
        try:
            value = _unpack_nested_value(element, self.attr_path)
        except Exception as E:
            raise KeyError(f"Attempted to access [{' - '.join(self.attr_path)}] "
                           f"on an object of <{element.__class__.__name__}>. "
                           f"Failed with {E}")
        result = value not in self.exclude
        if self.include:
            result = result and (value in self.include)
        return result


@dataclass
class ComparisonFilter(FilterBase):

    attr_path: List[str]
    _comparison_operators = {
        op: getattr(operator, op) for op in "gt,lt,ge,le,eq,ne".split(",")
    }
    gt: Optional[float] = None
    lt: Optional[float] = None
    ge: Optional[float] = None
    le: Optional[float] = None
    eq: Optional[float] = None
    ne: Optional[float] = None

    def __call__(self, element: Sample) -> bool:
        try:
            value = _unpack_nested_value(element, self.attr_path)
        except Exception as E:
            raise KeyError(f"Attempted to access [{' - '.join(self.attr_path)}] "
                           f"on an object of <{element.__class__.__name__}>. "
                           f"Failed with {E}")
        passing = []
        for op_name, op in self._comparison_operators.items():
            comparison_base = getattr(self, op_name)
            if comparison_base is None:
                continue
            passing.append(op(value, comparison_base))
        return all(passing)


_all_filters = {
    "comparison": ComparisonFilter,
    "inclusionexclusion": InclusionExclusionFilter
}


def factory(spec: List[dict]) -> List[FilterBase]:
    result = []
    for filter_spec in spec:
        filter_name = filter_spec.pop("name").lower()
        filter_type = _all_filters.get(filter_name)
        if filter_type is None:
            raise NameError(f"Unknown filter type: {filter_name}")
        result.append(filter_type(**filter_spec))
    return result
