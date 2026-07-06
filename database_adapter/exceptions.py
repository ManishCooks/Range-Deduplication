"""
Adapter Exceptions.
"""


class AdapterError(Exception):
    """Base adapter error."""
    pass


class ConnectionError(AdapterError):
    """Connection failed."""
    pass


class QueryError(AdapterError):
    """Query failed."""
    pass


class InsertError(AdapterError):
    """Insert failed."""
    pass
