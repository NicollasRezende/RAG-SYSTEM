"""
Módulo principal do pipeline RAG (Retrieval-Augmented Generation).
"""
from typing import List, Optional, Dict, Any
import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import Document

from vector_store import VectorStore


class RAGPipeline:
    """Pipeline completo de RAG para geração aumentada por recuperação."""

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
    ):
        """
        Inicializa o pipeline RAG com Ollama.

        Args:
            vector_store: Instância do VectorStore
            model_name: Nome do modelo Ollama (llama3.2, mistral, gemma2, etc)
            temperature: Temperatura para geração
            base_url: URL do servidor Ollama
        """
        self.vector_store = vector_store
        self.temperature = temperature
        self.model_name = model_name

        # Inicializa o modelo LLM com Ollama
        print(f"Conectando ao Ollama (modelo: {model_name})...")
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )
        print(f"✓ Modelo {model_name} carregado!")

        # Template de prompt padrão em português
        self.default_prompt_template = """Você é um assistente útil que responde perguntas com base no contexto fornecido.

Contexto relevante:
{context}

Pergunta: {question}

Instruções:
- Responda a pergunta com base APENAS nas informações fornecidas no contexto acima
- Se a informação não estiver no contexto, diga que você não tem informação suficiente para responder
- Seja claro, conciso e objetivo
- Use exemplos do contexto quando relevante

Resposta:"""

        self.qa_chain = None

    def setup_qa_chain(
        self,
        prompt_template: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Configura a chain de Q&A.

        Args:
            prompt_template: Template customizado de prompt
            search_kwargs: Argumentos para a busca (ex: {'k': 4})
        """
        search_kwargs = search_kwargs or {'k': 4}

        # Cria o retriever
        retriever = self.vector_store.get_retriever(search_kwargs=search_kwargs)

        # Usa template customizado ou padrão
        template = prompt_template or self.default_prompt_template

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Cria a chain de Q&A
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        print("QA Chain configurada com sucesso!")

    def query(
        self,
        question: str,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Faz uma pergunta ao sistema RAG.

        Args:
            question: Pergunta a ser respondida
            return_sources: Se deve retornar os documentos fonte

        Returns:
            Dicionário com a resposta e opcionalmente os documentos fonte
        """
        if self.qa_chain is None:
            raise ValueError("QA Chain não configurada. Execute setup_qa_chain() primeiro.")

        print(f"\nProcessando pergunta: {question}")

        result = self.qa_chain.invoke({"query": question})

        response = {
            "question": question,
            "answer": result["result"]
        }

        if return_sources and "source_documents" in result:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]

        return response

    def simple_retrieval(
        self,
        query: str,
        k: int = 4,
        with_scores: bool = False
    ) -> List[Document]:
        """
        Faz uma busca simples sem geração.

        Args:
            query: Texto da consulta
            k: Número de documentos a retornar
            with_scores: Se deve incluir scores de relevância

        Returns:
            Lista de documentos relevantes
        """
        if with_scores:
            return self.vector_store.similarity_search_with_score(query, k=k)
        else:
            return self.vector_store.similarity_search(query, k=k)

    def custom_query(
        self,
        question: str,
        custom_prompt: str,
        k: int = 4
    ) -> str:
        """
        Faz uma consulta com prompt customizado.

        Args:
            question: Pergunta a ser respondida
            custom_prompt: Template de prompt customizado
            k: Número de documentos a recuperar

        Returns:
            Resposta gerada
        """
        # Recupera documentos relevantes
        docs = self.vector_store.similarity_search(question, k=k)

        # Combina o contexto
        context = "\n\n".join([doc.page_content for doc in docs])

        # Formata o prompt
        formatted_prompt = custom_prompt.format(
            context=context,
            question=question
        )

        # Gera resposta
        response = self.llm.invoke(formatted_prompt)

        return response.content
