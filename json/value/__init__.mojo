# json - `Value` package
#
# This package replaces the v0.1 monolithic `json/value.mojo`. The split
# isolates pure string utilities (`raw_ops.mojo`) from the type itself
# (`value.mojo`) so subsequent phases can land focused changes:
#   - Phase B: copy-on-write mutation through `OwnedValue` (owned.mojo).
#   - Phase C: tape-backed read path via `Document` (json/document.mojo).
#
# Public re-exports below match the v0.1 surface so callers
# (patch, jsonpath, schema, serialize, reflection, lazy) build
# unchanged.

from .value import (
    Value,
    Null,
    make_view_value,
    _value_to_json,
)
from .raw_ops import (
    _extract_field_value,
    _extract_array_element,
    _extract_json_value,
    _count_array_elements,
    _extract_object_keys,
    _parse_json_pointer,
)
from .owned import (
    OwnedValue,
    _value_to_owned,
    _parse_owned_value,
    _owned_to_json,
    _materialize_for_write,
    _serialize_into_value,
    _set_at_pointer,
)
