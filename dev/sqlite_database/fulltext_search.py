from sqlalchemy import create_engine, Column, Integer, String, text, event, bindparam
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError

engine = create_engine("sqlite:///fulltext_search.db")
Base = declarative_base()

Session = sessionmaker(bind=engine)


class Document(Base):
    """A dummy table with documents on which we want to have fast full text search"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)

    @staticmethod
    def fulltext_search(search_query: str):
        """
        Perform a full-text search on the documents table.

        Args:
            search_query (str): The search query to match against the documents.

        Returns:
            list: A list of documents that match the search query.

        """
        with Session() as session:
            query = text(
                "SELECT * FROM documents_fts WHERE documents_fts MATCH :search_term"
            )
            results = session.execute(query, {"search_term": search_query}).fetchall()
        return results

    @staticmethod
    def bm25_fulltext_search(search_query: str):
        """
        Perform a full-text search on the documents table. Order results by bm25 score.

        Args:
            search_query (str): The search query to match against the documents.

        Returns:
            list: A list of documents that match the search query.
        """
        with Session() as session:
            query = text(
                "SELECT * FROM documents_fts WHERE fts MATCH :search_term ORDER BY bm25(fts)"
            )
            results = session.execute(query, {"search_term": search_query}).fetchall()
        return results


@event.listens_for(Document.__table__, "after_create")
def create_fts_table(_element, connection, **_kw):
    """
    Create the full-text search table 'documents_fts'.
    Triggered whenever the original documents table is created

    Args:
        _element: unused
        connection: connection to create the table
        _kw: unused
    """
    try:
        connection.execute(
            text("CREATE VIRTUAL TABLE documents_fts USING fts5(title, content)")
        )
    except OperationalError:
        pass


@event.listens_for(Document.__table__, "after_drop")
def drop_fts_table(_element, connection, **_kw):
    """
    Drop the full-text search table 'documents_fts' if it exists.
    Triggered whenever the original documents table is dropped

    Args:
        _element: unused
        connection: connection to drop the table
        _kw: unused
    """
    try:
        connection.execute(text("DROP TABLE IF EXISTS documents_fts"))
    except OperationalError:
        pass


@event.listens_for(Document, "after_insert", propagate=True)
def insert_into_fts5(_mapper, connection, target):
    """
    Inserts a new document into the 'documents_fts' table after an insert operation
    on the 'documents' table.

    Args:
        _mapper: unused.
        connection: The database connection object.
        target: The newly inserted Document object.
    """
    insert_query = text(
        "INSERT INTO documents_fts (title, content) SELECT title, content FROM documents WHERE title=:title AND content=:content"
    ).bindparams(bindparam("title", String), bindparam("content", String))
    connection.execute(insert_query, {"title": target.title, "content": target.content})


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

with Session() as session:
    new_document = Document(
        title="Sample Document", content="This is the content of the document."
    )
    session.add(new_document)
    session.commit()
    new_document = Document(
        title="Second Sample Document", content="Hanna Banana go wild."
    )
    session.add(new_document)
    session.commit()

    results = Document.fulltext_search("Banana")
    print(results)
