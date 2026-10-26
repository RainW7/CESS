"""Read CESS parameter tables without relying on positional column guesses.

Supported layouts are source-by-column matrices, the historical
column-by-source-by-one arrays, and one-dimensional structured tables. Existing
HDF5 files are never modified. Matrix columns need ``attrs['name']`` (or explicit
``names``), and every layout is checked against the ID/source count.
"""

import numpy as np


def _names_tuple(names):
    if isinstance(names, (str, bytes, np.bytes_)):
        names = [names]
    names = np.asarray(names)
    if names.ndim != 1:
        raise ValueError("parameter column names must be one-dimensional")
    result = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in names
    )
    if not result or any(not name.strip() for name in result):
        raise ValueError("parameter column names must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError("parameter column names must be unique")
    return result


def _axis_order(dataset):
    order = getattr(dataset, "attrs", {}).get("axis_order")
    if order is None:
        return None
    if isinstance(order, (bytes, np.bytes_)):
        order = order.decode("utf-8")
    if isinstance(order, str):
        order = order.split(",")
    aliases = {"column": "field", "parameter": "field", "object": "source"}
    normalized = []
    for axis in order:
        if isinstance(axis, (bytes, np.bytes_)):
            axis = axis.decode("utf-8")
        axis = str(axis).strip().lower()
        normalized.append(aliases.get(axis, axis))
    return tuple(normalized)


class ParameterTable:
    """A validated source-by-column view; ``table[name]`` returns a column."""

    def __init__(self, values, names, layout, structured=None):
        self.values = values
        self.names = tuple(names)
        self.layout = layout
        self._structured = structured
        self._indices = {name: index for index, name in enumerate(self.names)}

    def __len__(self):
        return self.values.shape[0]

    def __contains__(self, name):
        return name in self._indices

    def __getitem__(self, name):
        if name not in self._indices:
            raise KeyError("parameter column {!r} is missing; available: {}".format(
                name, ", ".join(self.names)))
        if self._structured is not None:
            return self._structured[name]
        return self.values[:, self._indices[name]]


def read_parameter_table(dataset, ids=None, n_sources=None, names=None, required=()):
    """Return a :class:`ParameterTable` validated against the ID length.

    ``dataset`` may be an HDF5 dataset or a NumPy array. For arrays, pass the
    source count via ``ids`` or ``n_sources`` and pass names for plain matrices.
    HDF5 datasets can infer the count from their file's root ``ID`` dataset.
    Ambiguous square matrices require ``axis_order='source,field'`` metadata;
    conflicting names, shapes, or axis declarations raise ``ValueError``.
    """
    if ids is None and n_sources is None:
        file = getattr(dataset, "file", None)
        if file is not None and "ID" in file:
            ids = file["ID"]
    if ids is not None:
        count = len(ids)
        if n_sources is not None and n_sources != count:
            raise ValueError("n_sources does not match ID length")
        n_sources = count
    if n_sources is None or isinstance(n_sources, (bool, np.bool_)):
        raise ValueError("an ID length or explicit n_sources is required")
    if int(n_sources) != n_sources or n_sources < 0:
        raise ValueError("n_sources must be a non-negative integer")
    n_sources = int(n_sources)

    attrs = getattr(dataset, "attrs", {})
    metadata_names = attrs.get("name", attrs.get("columns"))
    if "name" in attrs and "columns" in attrs:
        if _names_tuple(attrs["name"]) != _names_tuple(attrs["columns"]):
            raise ValueError("name and columns metadata disagree")
    if names is not None:
        names = _names_tuple(names)
        if metadata_names is not None and names != _names_tuple(metadata_names):
            raise ValueError("explicit names disagree with dataset metadata")
    elif metadata_names is not None:
        names = _names_tuple(metadata_names)

    data = np.asarray(dataset)
    order = _axis_order(dataset)
    structured = None
    if data.dtype.names is not None:
        dtype_names = tuple(data.dtype.names)
        if names is not None and names != dtype_names:
            raise ValueError("structured dtype fields disagree with column metadata")
        names = dtype_names
        if data.shape != (n_sources,):
            raise ValueError("structured parameter table must have shape (N,)")
        if any(data[name].shape != (n_sources,) for name in names):
            raise ValueError("structured parameter columns must be scalar per source")
        if order is not None and order != ("source",):
            raise ValueError("axis_order disagrees with structured table")
        structured = data
        values = np.column_stack([data[name] for name in names])
        layout = "structured"
    else:
        if names is None:
            raise ValueError("matrix parameter tables require named columns")
        n_columns = len(names)
        if data.ndim == 2 and data.shape == (n_sources, n_columns):
            if order is not None and order != ("source", "field"):
                raise ValueError("axis_order disagrees with source-by-column matrix")
            if n_sources == n_columns and n_sources > 1 and order is None:
                raise ValueError("ambiguous square table: declare axis_order='source,field'")
            values = data
            layout = "source_by_column"
        elif data.ndim == 3 and data.shape == (n_columns, n_sources, 1):
            if order is not None and order != ("field", "source", "singleton"):
                raise ValueError("axis_order disagrees with legacy table")
            values = data[:, :, 0].T
            layout = "legacy_column_by_source_by_one"
        else:
            raise ValueError("parameter table shape {} does not match {} IDs and {} names".format(
                data.shape, n_sources, n_columns))
    table = ParameterTable(values, names, layout, structured=structured)
    if isinstance(required, str):
        required = (required,)
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError("required parameter columns are missing: {}".format(
            ", ".join(missing)))
    return table
