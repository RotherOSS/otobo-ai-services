import os
from pathlib import Path
import click
#from chromadb import HttpClient, PersistentClient, Settings, EmbeddingFunction, Embeddings, Documents
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from dotenv import load_dotenv
from langchain_chroma import Chroma
#from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from settings import AppSettings

settings = AppSettings()

console = Console()

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx,verbose):
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    pass

def get_embeddingsmodel():
    # Returns an embedding model instance using Ollama with config values
    return OpenAIEmbeddings(
        model=settings.OTOBO_AI_EMBEDDING_MODEL,
        base_url=f"{settings.OTOBO_AI_LLM_HOST}/v1",
        encoding_format= None,
        api_key=settings.OTOBO_AI_LLM_API_KEY or "ollama",
        check_embedding_ctx_length=False    
    )

def get_vectorstore(with_embedding: bool = True, collection_name: str = "faqs"):
    # Returns a Chroma vector store instance, optionally attaching an embedding function
    db_embedding = get_embeddingsmodel() if with_embedding else None

    return Chroma(
        collection_name=collection_name,
        embedding_function=db_embedding,        
        persist_directory="/chroma"  # Local dir for vector DB persistence
    )



@cli.command(name='list')
def list_collections():
    """List all collections"""
    client = get_vectorstore(with_embedding=False,collection_name="faqs")._client
    collections = client.list_collections()

    if not collections:
        console.print("[yellow]No collections found")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    # table.add_column("Distance")
    table.add_column("Count")

    for col in collections:
        collection = client.get_collection(col)
        table.add_row(
            col,
            # col.metadata.get("hnsw:space", "l2"),
            str(collection.count())
        )

    console.print(table)


@cli.command()
@click.argument('name')
@click.option('--limit', default=10, help='Number of items to show')
@click.option('--offset', default=0, help='Start index')
def peek(name: str, limit: int, offset: int):
    """Peek into a collection's contents"""
    collection = get_vectorstore(with_embedding=False,collection_name=name)
    try:
#        collection = vectorstore.get_collection(name)
        results = collection.get(limit=limit,offset=offset)

        if not results['ids']:
            console.print("[yellow]Collection is empty")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID")
        table.add_column("Metadata")
        table.add_column("Document Preview")

        for i in range(len(results['ids'])):

            doc_preview = results['documents'][i][:500] + "..." if len(
                results['documents'][i]) > 500 else results['documents'][i]
            #metadata = JSON(str(results['metadatas'][i])
            #                ) if results['metadatas'][i] else ""
            #table.add_row(results['ids'][i], metadata, doc_preview)
            metadata = str(results['metadatas'][i])
            table.add_row(results['ids'][i], metadata, doc_preview)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


@cli.command()
@click.argument('name')
@click.argument('query')
@click.option('--n-results', default=5, help='Number of results to return')
def search(name: str, query: str, n_results: int):
    """Search a collection with a text query"""
    collection = get_vectorstore(with_embedding=True,collection_name=name)

    try:
        results = collection.similarity_search(query=query, k=n_results)
        
        console.print(results)
 
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")


if __name__ == '__main__':
    cli()

