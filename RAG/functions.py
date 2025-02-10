import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get OpenAI configuration from environment
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_TEMPERATURE = float(os.environ.get('TEMPERATURE', '0'))

# Create a configured ChatOpenAI instance
def get_chat_openai():
    """Create a ChatOpenAI instance with configuration from environment variables"""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE
    )


def initialize_grader():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    class GradeDocuments(BaseModel):
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    llm = get_chat_openai()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader


def initialize_rag_chain():
    from langchain_core.output_parsers import StrOutputParser

    system = """
# Role
You are an AI assistant tasked with helping users answer questions about the "ЖизньМарт" franchise in the Russian language. Your goal is to analyze the provided data about the franchise, then answer the user's questions based on this data and explain it in Russian.

# Context:
## First, you will be provided with context information about the "ЖизньМарт" franchise.

# Task:
## Here are your instructions:

### 1. Carefully analyze the provided franchise context. Pay attention to all details, policies, terms, and any other relevant information.

### 2. Your task is to answer the user's question based on the information provided in the franchise context. Do not make assumptions or introduce information that is not explicitly stated in the context.

### 3. Structure your response as follows:
   a. Begin with a brief analysis of the relevant parts of the franchise context that pertain to the question.
   b. Provide a clear and concise answer to the question.
   c. Explain your reasoning, referencing specific elements from the franchise context to support your answer.

### 4. **It is crucial that your answers are correct. Do not make mistakes or provide inaccurate information under any circumstances.**

# Important Notes:
- Remember, your primary goal is to provide **accurate and correct** answers based on the given franchise context. **Correctness and accuracy are paramount**, and it's better to admit when you don't have enough information than to give an incorrect or speculative answer.
- Also **remember to use the Russian language** in your answer only!

        """

    generation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n Context: {context} \n\n Answer:"),
        ]
    )

    llm = get_chat_openai()
    return generation_prompt | llm | StrOutputParser()


def initialize_hallucination_grader():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    class GradeHallucinations(BaseModel):
        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    llm = get_chat_openai()
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. \n"""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader


def initialize_answer_grader():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    class GradeAnswer(BaseModel):
        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    llm = get_chat_openai()
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. \n"""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader


def initialize_question_rewriter():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    llm = get_chat_openai()

    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
         You must provide your answer in russian language."""

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()


# if __name__ == "__main__":
#     docs_list = load_docs()
#     doc_splits = split_documents(docs_list)
#     retriever = create_vectorstore(doc_splits)
#
#     retrieval_grader = initialize_grader()
#     rag_chain = initialize_rag_chain()
#     hallucination_grader = initialize_hallucination_grader()
#     answer_grader = initialize_answer_grader()
#     question_rewriter = initialize_question_rewriter()
#
#     question = "how to reduce llm costs"
#     docs = retriever.get_relevant_documents(question)
#     doc_txt = docs[1].page_content
#     print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
#
#     generation = rag_chain.invoke({"context": docs, "question": question})
#     print(generation)
#
#     print(hallucination_grader.invoke({"documents": docs, "generation": generation}))
#     print(answer_grader.invoke({"question": question, "generation": generation}))
#     print(question_rewriter.invoke({"question": question}))