"""Extra read-only checks for PostgreSQL queries."""

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


class UnsafeSQLError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def ensure_read_only(sql: str, allowed_tables: frozenset[str], max_rows: int = 500) -> str:
    """Allow listed tables unqualified or in the public schema only."""
    try:
        statements = sqlglot.parse(sql, read="postgres")
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
            raise UnsafeSQLError("Only one SELECT query is allowed.")
        tree = statements[0]
        forbidden = (exp.DML, exp.DDL, exp.Command, exp.Into, exp.Lock)
        if any(isinstance(node, forbidden) for node in tree.walk()):
            raise UnsafeSQLError("Only read-only queries are allowed.")
        for node in tree.find_all(exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in {"pg_sleep", "set_config", "current_setting"} or name.startswith(
                ("pg_read_", "lo_", "dblink")
            ):
                raise UnsafeSQLError("This function is not allowed.")
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    if (
                        source.catalog
                        or (source.db and source.db != "public")
                        or source.name not in allowed_tables
                    ):
                        raise UnsafeSQLError("This table is not allowed.")
        limit = tree.args.get("limit")
        value = limit.expression if limit else None
        if not isinstance(value, exp.Literal) or not value.is_int or int(value.this) > max_rows:
            tree = tree.limit(max_rows, copy=False)
        return tree.sql(dialect="postgres")
    except UnsafeSQLError:
        raise
    except Exception as exc:
        raise UnsafeSQLError("The query could not be safely parsed.") from exc
