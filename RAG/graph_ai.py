import logging
from RAG.functions import (
    initialize_grader,
    initialize_rag_chain,
    initialize_hallucination_grader,
    initialize_answer_grader,
    initialize_question_rewriter,
)
from typing_extensions import TypedDict
from typing import List
from openai_helper import OpenAIHelper
from RAG.rag import EmbeddingService

# Configure logging
logger = logging.getLogger(__name__)

# Инициализация всех компонентов
retrieval_grader = initialize_grader()
rag_chain = initialize_rag_chain()
hallucination_grader = initialize_hallucination_grader()
answer_grader = initialize_answer_grader()
question_rewriter = initialize_question_rewriter()

# Initialize EmbeddingService (Singleton pattern ensures single instance)
embedding_service = EmbeddingService()
logger.info("EmbeddingService initialized successfully")


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        transformation_count: количество перегенераций вопроса
        first_question: первый вопрос пользователя
    """
    question: str
    generation: str
    documents: List[str]
    transformation_count: int
    first_question: str
    openai_helper: OpenAIHelper
    total_tokens: int
    chat_id: int


### Nodes

def retrieve(state: GraphState):
    logger.info("---RETRIEVE---")
    question = state["question"]
    
    # Use EmbeddingService's fusion_retrieval for better search results
    documents = embedding_service.fusion_retrieval(
        query=question,
        k=5,  # Number of most relevant documents to return
        alpha=0.5  # Balance between semantic (0.5) and keyword search (0.5)
    )
    
    first_question = state.get("first_question", question)
    logger.info(f"Retrieved {len(documents)} relevant documents for question: {question}")
    
    return {
        "documents": documents,
        "question": question,
        "transformation_count": state.get("transformation_count", 0),
        "first_question": first_question,
        "openai_helper": state["openai_helper"],
        "total_tokens": state["total_tokens"],
        "chat_id": state["chat_id"]
    }


async def generate(state: GraphState):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    openai_helper = state["openai_helper"]
    chat_id = state["chat_id"]

    prompt = f"""You are an assistant for question-answering tasks.\n
         Use the following pieces of retrieved context to answer the question.\n
         **If you don't know the answer, just say that you don't know.** \n
         Keep the answer concise. \n\n
         User question: \n\n {question} \n\n Context: {documents} \n\n Answer:"""
    total_tokens = 0
    generation, total_tokens = await openai_helper.get_chat_response(chat_id=chat_id, query=prompt)
    print('\n')
    print(generation)
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "transformation_count": state.get("transformation_count", 0),
        "first_question": state["first_question"],
        "openai_helper": state["openai_helper"],
        "total_tokens": total_tokens,
        "chat_id": state["chat_id"]
    }


def grade_documents(state: GraphState):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            print(f"\n{d}\n")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            print(f"\n{d}\n")
            continue
        print('\n')
        print(filtered_docs)
    return {
        "documents": filtered_docs,
        "question": question,
        "transformation_count": state.get("transformation_count", 0),
        "first_question": state["first_question"],
        "openai_helper": state["openai_helper"],
        "total_tokens": state["total_tokens"],
        "chat_id": state["chat_id"]
    }


def transform_query(state: GraphState):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    print('\n')
    print(better_question)
    return {
        "documents": documents,
        "question": better_question,
        "transformation_count": state.get("transformation_count", 0),
        "first_question": state["first_question"],
        "openai_helper": state["openai_helper"],
        "total_tokens": state["total_tokens"],
        "chat_id": state["chat_id"]
    }


def transformation_count_increment(state: GraphState):
    cur_transformation_count = state.get("transformation_count", 0)

    cur_transformation_count += 1

    return {
        "documents": state["documents"],
        "question": state["question"],
        "transformation_count": cur_transformation_count,
        "first_question": state["first_question"],
        "openai_helper": state["openai_helper"],
        "total_tokens": state["total_tokens"],
        "chat_id": state["chat_id"]
    }


### Edges

def decide_to_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    filtered_documents = state["documents"]

    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, SENDING SORRY MESSAGE---")
        return "send_sorry_message"
    else:
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_vs_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    first_question = state["first_question"]
    print('\n')
    print(question)
    print('\n')

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTС---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": first_question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def decide_to_transform_query(state: GraphState):
    print("---DECIDING WHETHER TO TRANSFORM QUERY OR NOT---")

    cur_transformation_count = state.get("transformation_count", 0)
    if cur_transformation_count is None:
        cur_transformation_count = 0

    if cur_transformation_count > 1:
        print("---DECISION: cur_transformation_count >= 1 => send_sorry_message---")
        return "send_sorry_message"

    return "transform_query"


def send_sorry_message(state: GraphState):
    print("---NO RELEVANT DOCUMENTS FOUND---")
    return {"question": state["question"],
            "documents": state["documents"],
            "generation": "К сожалению, в базе знаний не нашлось релевантной информации, чтобы ответить на ваш вопрос.\n"
                          "Если вы хотите получить точный ответ на этот специфический запрос, обратитесь в службу поддержки"
                          "по телефону: +7 ... или переформулируйте ваш вопрос",
            "transformation_count": state["transformation_count"],
            "first_question": state["first_question"],
            "openai_helper": state["openai_helper"],
            "total_tokens": state["total_tokens"],
            "chat_id": state["chat_id"]
            }
