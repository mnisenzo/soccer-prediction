"""Database engine and session management."""
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from .schema import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None

# Default DB path relative to the project root
_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "soccer.db"


def get_engine(db_url: Optional[str] = None) -> Engine:
    global _engine
    if _engine is None:
        if db_url is None:
            _DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{_DEFAULT_DB_PATH}"
        _engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    return _engine


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_url)
        _SessionLocal = sessionmaker(bind=engine, autoflush=True, autocommit=False)
    return _SessionLocal


def init_db(db_url: Optional[str] = None) -> None:
    """Create all tables. Safe to call multiple times."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """Force a new engine on next call (useful for testing with a temp DB)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def get_session(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    factory = get_session_factory(db_url)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
