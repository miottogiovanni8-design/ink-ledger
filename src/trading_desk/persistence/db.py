import os
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from trading_desk.persistence.models import Base


def make_engine(db_path: str):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def _sql_literal(value):
    """A Python scalar default as a SQLite DDL literal, or None if it
    isn't one (e.g. a callable like utcnow) — those get added as a
    nullable column instead of failing the migration outright."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _sync_missing_columns(engine) -> None:
    """create_all only creates missing tables, never adds columns to a
    table that already exists — so a column added to a model after the
    live db file was first created silently never shows up there, and
    the first insert touching it fails deep inside a flush at runtime
    instead of here. Adds whatever columns the models declare that the
    actual file doesn't have yet, one ALTER TABLE ADD COLUMN at a time
    (SQLite doesn't support altering several columns in one statement).

    mapped_column(default=...) is an ORM-side default, not a DB-level
    one, so it never shows up in the column's compiled DDL — and SQLite
    refuses to ADD COLUMN ... NOT NULL on a non-empty table without a
    real DEFAULT. So the literal default is recovered here by hand where
    it's a simple scalar; anything else (e.g. a callable like utcnow)
    gets added as nullable instead of failing the migration outright."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                literal = None
                if column.default is not None and column.default.is_scalar:
                    literal = _sql_literal(column.default.arg)
                ddl = f"{column.name} {col_type}"
                if literal is not None:
                    ddl += f" NOT NULL DEFAULT {literal}" if not column.nullable else f" DEFAULT {literal}"
                conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {ddl}")


def init_db(db_path: str) -> None:
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    _sync_missing_columns(engine)


@contextmanager
def get_session(db_path: str):
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    _sync_missing_columns(engine)
    factory = sessionmaker(bind=engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
