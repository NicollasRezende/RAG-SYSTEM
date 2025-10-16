"""
Módulo para carregar e processar diferentes tipos de documentos.
"""
import os
from typing import List
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentLoader:
    """Carrega e processa documentos de diferentes formatos."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa o carregador de documentos.

        Args:
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_single_file(self, file_path: str) -> List[Document]:
        """
        Carrega um único arquivo.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Lista de documentos
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Seleciona o loader apropriado baseado na extensão
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(f"Tipo de arquivo não suportado: {file_path.suffix}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """
        Carrega todos os arquivos de um diretório.

        Args:
            directory_path: Caminho do diretório
            glob_pattern: Padrão glob para filtrar arquivos

        Returns:
            Lista de documentos
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

        all_documents = []

        # Carrega arquivos TXT e MD
        txt_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        all_documents.extend(txt_loader.load())

        md_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.md",
            loader_cls=TextLoader
        )
        all_documents.extend(md_loader.load())

        # Carrega PDFs
        pdf_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        all_documents.extend(pdf_loader.load())

        # Carrega documentos Word
        docx_loader = DirectoryLoader(
            str(directory_path),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader
        )
        all_documents.extend(docx_loader.load())

        # Divide em chunks
        return self.text_splitter.split_documents(all_documents)

    def load_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Carrega texto direto.

        Args:
            text: Texto a ser carregado
            metadata: Metadados opcionais

        Returns:
            Lista de documentos
        """
        document = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([document])
