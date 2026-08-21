import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_desk.persistence.models import Base


def make_engine(db_path: str):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    return create_engine(f"sqlite:///{db_path}")


def init_db(db_path: str) -> None:
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)


@contextmanager
def get_session(db_path: str):
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
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
