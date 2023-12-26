from langchain.chat_models import ChatOpneAi
from langchain.prompt.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from vectorDB import vectordb
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "config/system.txt")
INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "config/intent_list.txt")
CHECK_INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "config/check_intent.txt")
GET_ANSWER_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "config/get_answer.txt")

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        template=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )
    
check_intent_chain = create_chain(
    llm=llm,
    template_path=CHECK_INTENT_PROMPT_TEMPLATE,
    output_key="intent"
)

retriever = vectordb.get[context["intent"]].as_retriever()
docs = retriever.get_relavant_documents(query, search_type='similarity', search_kwargs={'k': 5})
    
def generate_chain(llm):
    chains = {
        "check_intent": create_chain(
            llm=llm,
            template_path=CHECK_INTENT_PROMPT_TEMPLATE,
            output_key="intent",
        ),
        "get_related_text": create_chain(
        
        ),
        "get_answer": create_chain(
            llm=llm,
            template_path=GET_ANSWER_PROMPT_TEMPLATE,
            output_key="answer"
        )
    }