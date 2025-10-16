"""
Script principal de exemplo para usar o sistema RAG.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Adiciona o diretório src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline


def main():
    """Função principal de exemplo."""

    # Carrega variáveis de ambiente
    load_dotenv()

    print("=" * 60)
    print("Sistema RAG - Retrieval-Augmented Generation")
    print("=" * 60)

    # Configuração
    DATA_DIR = "./data"
    CHROMA_DIR = "./chroma_db"

    # 1. Carrega documentos
    print("\n[1] Carregando documentos...")
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)

    # Verifica se existe diretório de dados
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        print(f"\nCriando diretório de dados: {DATA_DIR}")
        data_path.mkdir(parents=True, exist_ok=True)

        # Cria um documento de exemplo
        example_file = data_path / "exemplo.txt"
        with open(example_file, "w", encoding="utf-8") as f:
            f.write("""
# Documentação do Sistema RAG

## O que é RAG?
RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informações
com geração de texto usando modelos de linguagem. O sistema primeiro busca documentos
relevantes em uma base de conhecimento e depois usa essas informações para gerar
respostas contextualizadas.

## Componentes principais
1. Document Loader: Carrega e processa documentos de diferentes formatos
2. Vector Store: Armazena embeddings dos documentos para busca eficiente
3. Retriever: Busca documentos relevantes baseado em similaridade semântica
4. LLM: Gera respostas baseadas nos documentos recuperados

## Vantagens do RAG
- Respostas baseadas em fontes específicas de conhecimento
- Reduz alucinações do modelo
- Permite atualização da base de conhecimento sem re-treinar o modelo
- Transparência através das fontes citadas
            """)
        print(f"Arquivo de exemplo criado: {example_file}")

    # Carrega documentos do diretório
    try:
        documents = loader.load_directory(DATA_DIR)
        print(f"✓ {len(documents)} chunks de documentos carregados")
    except Exception as e:
        print(f"Erro ao carregar documentos: {e}")
        return

    # 2. Cria/carrega vector store
    print("\n[2] Configurando Vector Store...")
    vector_store = VectorStore(
        persist_directory=CHROMA_DIR,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Verifica se já existe um vector store
    if Path(CHROMA_DIR).exists():
        try:
            vector_store.load_vectorstore()
            print("✓ Vector Store carregado do disco")

            # Pergunta se quer adicionar novos documentos
            response = input("\nDeseja adicionar novos documentos? (s/n): ")
            if response.lower() == 's':
                vector_store.add_documents(documents)
        except Exception as e:
            print(f"Erro ao carregar vector store: {e}")
            print("Criando novo vector store...")
            vector_store.create_vectorstore(documents)
    else:
        vector_store.create_vectorstore(documents)
        print("✓ Vector Store criado com sucesso")

    # 3. Configura pipeline RAG com Ollama
    print("\n[3] Configurando Pipeline RAG com Ollama...")

    # Lista modelos instalados
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed_models = []
        for line in result.stdout.split('\n')[1:]:  # Pula header
            if line.strip():
                model_name_full = line.split()[0]
                installed_models.append(model_name_full)

        if installed_models:
            print(f"Modelos instalados: {', '.join(installed_models)}")
            default_model = installed_models[0]
        else:
            print("Nenhum modelo encontrado. Sugestão: ollama pull qwen2.5:1.5b")
            default_model = "qwen2.5:1.5b"
    except:
        print("Modelos sugeridos: qwen2.5:1.5b, llama3.2, mistral")
        default_model = "qwen2.5:1.5b"

    # Permite escolher o modelo
    model_choice = input(f"\nEscolha o modelo (Enter para {default_model}): ").strip()
    model_name = model_choice if model_choice else default_model

    try:
        # Cria pipeline RAG
        rag = RAGPipeline(vector_store=vector_store, model_name=model_name)
        rag.setup_qa_chain(search_kwargs={'k': 3})
        print("✓ Pipeline RAG configurado")
    except Exception as e:
        print(f"\n❌ Erro ao conectar com Ollama: {e}")
        print("\nVerifique se:")
        print("1. Ollama está instalado: https://ollama.ai")
        print("2. Ollama está rodando: ollama serve")
        print(f"3. O modelo está baixado: ollama pull {model_name}")
        return

    # 4. Interface de perguntas
    print("\n" + "=" * 60)
    print("Sistema RAG Pronto! Digite suas perguntas (ou 'sair' para encerrar)")
    print("=" * 60)

    while True:
        print("\n" + "-" * 60)
        question = input("\n💬 Pergunta: ").strip()

        if question.lower() in ['sair', 'exit', 'quit', 'q']:
            print("\nEncerrando sistema RAG. Até logo!")
            break

        if not question:
            continue

        try:
            # Faz a consulta
            result = rag.query(question, return_sources=True)

            # Exibe a resposta
            print(f"\n🤖 Resposta:\n{result['answer']}")

            # Exibe as fontes
            if 'sources' in result and result['sources']:
                print(f"\n📚 Fontes utilizadas ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n  [{i}] {source['metadata'].get('source', 'N/A')}")
                    print(f"      {source['content'][:150]}...")

        except Exception as e:
            print(f"\n❌ Erro ao processar pergunta: {e}")


if __name__ == "__main__":
    main()
