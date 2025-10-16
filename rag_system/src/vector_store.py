"""
Módulo para gerenciar embeddings e vector store usando ChromaDB.
"""
from typing import List, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStore:
    """Gerencia embeddings e armazenamento vetorial."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "rag_collection"
    ):
        """
        Inicializa o vector store.

        Args:
            persist_directory: Diretório para persistir o banco de dados
            embedding_model: Modelo de embeddings do HuggingFace
            collection_name: Nome da coleção no ChromaDB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Inicializa o modelo de embeddings
        print(f"Carregando modelo de embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vectorstore: Optional[Chroma] = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Cria um novo vector store a partir de documentos.

        Args:
            documents: Lista de documentos

        Returns:
            Vector store criado
        """
        print(f"Criando vector store com {len(documents)} documentos...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        print(f"Vector store criado e persistido em: {self.persist_directory}")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Carrega um vector store existente.

        Returns:
            Vector store carregado
        """
        persist_path = Path(self.persist_directory)

        if not persist_path.exists():
            raise FileNotFoundError(
                f"Vector store não encontrado em: {self.persist_directory}"
            )

        print(f"Carregando vector store de: {self.persist_directory}")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

        return self.vectorstore

    def add_documents(self, documents: List[Document]):
        """
        Adiciona documentos ao vector store existente.

        Args:
            documents: Lista de documentos a adicionar
        """
        if self.vectorstore is None:
            raise ValueError("Vector store não inicializado. Use create_vectorstore() ou load_vectorstore() primeiro.")

        print(f"Adicionando {len(documents)} documentos ao vector store...")
        self.vectorstore.add_documents(documents)
        print("Documentos adicionados com sucesso!")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Busca documentos similares à query.

        Args:
            query: Texto da consulta
            k: Número de documentos a retornar
            filter_metadata: Filtros opcionais de metadados

        Returns:
            Lista de documentos mais similares
        """
        if self.vectorstore is None:
            raise ValueError("Vector store não inicializado.")

        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter_metadata
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        Busca documentos similares com scores de relevância.

        Args:
            query: Texto da consulta
            k: Número de documentos a retornar

        Returns:
            Lista de tuplas (documento, score)
        """
        if self.vectorstore is None:
            raise ValueError("Vector store não inicializado.")

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self):
        """Remove a coleção do vector store."""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print(f"Coleção '{self.collection_name}' removida.")

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Retorna um retriever para uso com chains do LangChain.

        Args:
            search_kwargs: Argumentos de busca (ex: {'k': 4})

        Returns:
            Retriever configurado
        """
        if self.vectorstore is None:
            raise ValueError("Vector store não inicializado.")

        search_kwargs = search_kwargs or {'k': 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
