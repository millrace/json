# json - `Value` package
#
# This package replaces the v0.1 monolithic `json/value.mojo`. The split
# isolates pure string utilities (`raw_ops.mojo`) from the type itself
# (`value.mojo`) so subsequent phases can land focused changes:
#   - Phase B: copy-on-write mutation through `OwnedValue`.
#   - Phase C: tape-backed read path via `Document` (json/document.mojo).
#
# Public re-exports below match the v0.1 surface so callers
# (patch, jsonpath, schema, serialize, reflection, lazy) build
# unchanged.

from .value import (
    Value,
    Null,
    make_array_value,
    make_object_value,
    _value_to_json,
    _parse_json_value_to_value,
    _navigate_pointer,
)
from .raw_ops import (
    _extract_field_value,
    _extract_array_element,
    _extract_json_value,
    _count_array_elements,
    _extract_object_keys,
    _update_object_value,
    _add_object_key,
    _update_array_element,
    _append_to_array,
    _find_value_end_str,
    _parse_json_pointer,
)
