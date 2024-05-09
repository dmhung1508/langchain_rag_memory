
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import cfg,os
import torch
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import RetrievalQAWithSourcesChain
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from langchain_community.document_loaders import JSONLoader
from fastapi import FastAPI, File, UploadFile
from typing import Optional
from pyngrok import ngrok
import uvicorn
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

CONDENSE_QUESTION_PROMPT.template

llm = ChatGroq(
    temperature=0,
    model_name= "llama3-8b-8192",
    groq_api_key= "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",
    max_tokens = 1025,
)


if torch.cuda.is_available():
    device = 'cuda'
    gpu_layers = cfg.gpu_layers
elif torch.backends.mps.is_available():
    device = 'mps'
    gpu_layers = 1
else:
    device = 'cpu'
    gpu_layers = 0
model_name = cfg.embedding  # Using open source embedding model

embedding_function = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True} #normalizes the vectors
)

print(f'Embedding Model loaded: {model_name}')


client = QdrantClient(
    url=cfg.QDRANT_URL,
    api_key=cfg.QDRANT_API_KEY,
    prefer_grpc=False
)


db = Qdrant(client=client, embeddings=embedding_function, collection_name=cfg.QDRANT_COLLECTION_NAME)

retriever = db.as_retriever(search_kwargs={"k": 3})



# Need a new default prompt that includes the summaries (the data retrieved by RAG)
default_prompt_with_context = cfg.default_prompt_with_context

chain_type_kwargs={
        'prompt': PromptTemplate(
            template=default_prompt_with_context,
            input_variables=['summaries', 'question'],
        ),
    }

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type='stuff', # stuff means that the context is "stuffed" into the context
    retriever=retriever,
    return_source_documents=True, # This returns the sources used by RAG
    chain_type_kwargs=chain_type_kwargs
)
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["group"] = record.get("group")
    metadata["question"] = record.get("question")

    return metadata




def loada(path):
    loader = JSONLoader(
        file_path=path,
        jq_schema='.[]',
        content_key="answer",
        metadata_func=metadata_func
    )

    data = loader.load()
    return data

def update(path):
    global retriever
    data = loada(path)
    qdrant = Qdrant.from_documents(
        data,
        embedding_function,
        url=cfg.QDRANT_URL,
        prefer_grpc=False,
        collection_name=cfg.QDRANT_COLLECTION_NAME,
        api_key=cfg.QDRANT_API_KEY,
    )
    db = Qdrant(client=client, embeddings=embedding_function, collection_name=cfg.QDRANT_COLLECTION_NAME)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Update success")

def ra(text,time,source):
    variable_name = time

    # Kiểm tra xem biến có tồn tại trong hệ thống không
    if variable_name in globals():
        # Biến đã tồn tại, sử dụng nó
        variable_value = globals()[variable_name]
    else:
        # Biến chưa tồn tại, tạo mới nó
        variable_value = ConversationSummaryBufferMemory(
            llm=llm,
            input_key='question',
            output_key='answer',
            memory_key='chat_history',
            return_messages=True,
        )
        globals()[variable_name] = variable_value

    # Bây giờ bạn có thể sử dụng biến 'variable_value'
    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
    )

    answer_chain = load_qa_with_sources_chain(
        llm=llm,
        chain_type='stuff',
        verbose=False,
        prompt=PROMPT
    )

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=answer_chain,
        verbose=False,
        memory=variable_value,
        rephrase_question=True,
        return_source_documents=True,


    )
    result = chain({'question': text})

    return result





default_prompt = cfg.default_prompt
PROMPT = PromptTemplate(input_variables=['summaries', 'question'], template=default_prompt)









NGROK_STATIC_DOMAIN = cfg.NGROK_STATIC_DOMAIN
NGROK_TOKEN=          cfg.NGROK_TOKEN




origins = ["*"]
app_api = FastAPI()
app_api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app_api.get("/")
def read_root():
    return "API RAG"
@app_api.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    if not os.path.exists("file"):
        os.mkdir("file")
    
    contents = await file.read()
    with open(f"file/uploaded_{file.filename}", "wb") as f:
        f.write(contents)
    update(f"file/uploaded_{file.filename}")
    
    return JSONResponse(content={"filename": file.filename, "status": "file uploaded successfully"})
@app_api.get("/rag/")
async def read_item( time: str, q: Optional[str] = None, source: Optional[str] = None):
    if q:
        print(time)
        data = ra(q,time,source)
        sources = []
        for docs in data["source_documents"]:
            sources.append(docs.to_json()["kwargs"])
        res = {
            "result" : data["answer"],
            "source_documents":sources
        }
        return JSONResponse(content=jsonable_encoder(res))
    return None

ngrok.set_auth_token(NGROK_TOKEN)
ngrok_tunnel = ngrok.connect(4501,domain=NGROK_STATIC_DOMAIN)
print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
uvicorn.run(app_api, port=4501)
