from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

social_loader = DirectoryLoader(DATA_DIR+"/project_data_카카오소셜.txt")
sync_loader = DirectoryLoader(DATA_DIR+"/project_data_카카오싱크.txt")
channel_loader = DirectoryLoader(DATA_DIR+"/project_data_카카오톡채널.txt")

social_documents = social_loader.load()
sync_documents = sync_loader.load()
channel_documents = channel_loader.load()

social_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
sync_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
channel_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

social_texts = social_splitter.split_documents(social_documents)
sync_texts = sync_splitter.split_documents(sync_documents)
channel_texts = channel_splitter.split_documents(channel_documents)

social_persist_dir = 'social_db'
sync_persist_dir = 'sync_db'
channel_persist_dir = 'channel_db'

embeddings = OpenAIEmbeddings()

social_vectordb = Chroma.from_documents(
    documents=social_texts,
    embeddings=embeddings,
    persist_directory=social_persist_dir
)

sync_vectordb = Chroma.from_documents(
    documents=sync_texts,
    embeddings=embeddings,
    persist_directory=sync_persist_dir
)

channel_vectordb = Chroma.from_documents(
    documents=channel_texts,
    embeddings=embeddings,
    persist_directory=channel_persist_dir
)

social_vectordb.persist()
sync_vectordb.persist()
channel_vectordb.persist()

social_vectordb = None
sync_vectordb = None
channel_vectordb = None

social_vectordb = Chroma(
    persist_directory=social_persist_dir,
    embedding_function=embeddings
)
sync_vectordb = Chroma(
    persist_directory=sync_persist_dir,
    embedding_function=embeddings
)
channel_vectordb = Chroma(
    persist_directory=channel_persist_dir,
    embedding_function=embeddings
)

class vectordb:
    social = social_vectordb
    sync = sync_vectordb
    channel = channel_vectordb