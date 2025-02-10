

from RAG.rag import EmbeddingService
from graph_state import GraphState, ClientQualification

embedding_service = EmbeddingService()

prompt_to_gpt = """
Опираясь свои профессиональные знания из контекста, ответь пользователю на вопрос:
{question}

"""

prompt_to_gpt_second_agent = """
Опираясь на свои инструкции, ответь клиенту:\n 
{question}
"""

def branch_logic(state):
    # Загружаем беседу
    conversations = state.openai_helper.conversations.get(state.chat_id, [])
    # Извлекаем первое системное сообщение
    first_system_message = conversations[0]
    system_message_text = first_system_message["content"]
    system_message_text = system_message_text.replace("{{DATA}}", str(embedding_service.fusion_retrieval(state.question)))
    conversations[0]["content"] = system_message_text


# Функция для GPT генерации
async def gpt(state: GraphState):
    question = state.question
    chat_id = state.chat_id
    openai_helper = state.openai_helper

    query = prompt_to_gpt.format(question=question)

    branch_logic(state)
    generation, total_tokens = await openai_helper.get_chat_response(chat_id=chat_id, query=query)

    state.generation = generation  # Обновляем состояние
    state.total_tokens += int(total_tokens)
    return state


# async def gpt_second_agent(state: GraphState):
#    question = state.question
#    chat_id = state.chat_id
#    openai_helper = state.openai_helper
#
#    # Обновляем сообщение с ролью 'system' в истории чата
#    for message in openai_helper.conversations.get(chat_id, []):
#        if message["role"] == "system":
#            message["content"] = system_prompt_to_gpt_second_agent
#            break  # Обновляем только одно сообщение и выходим из цикла
#
#    query = prompt_to_gpt_second_agent.format(question=question)
#
#    generation, total_tokens = await openai_helper.get_chat_response(chat_id=chat_id, query=query)
#
#    # Обновляем состояние
#    state.generation = generation
#    state.total_tokens += total_tokens
#    return state



def save_qualification(state: GraphState, qualification: ClientQualification):
    if qualification:
        qualification_data = qualification.dict()
        if state.qualification:
            for field, value in qualification_data.items():
                if value is not None and getattr(state.qualification, field) is None:
                    setattr(state.qualification, field, value)
        else:
            state.qualification = qualification
        print("Qualification data updated:", state.qualification)
    else:
        print("No qualification data to save.")
    return state


async def run_graph(state: GraphState):
    state = await gpt(state)

    # Результаты
    final_generation = state.generation
    total_tokens = state.total_tokens

    return final_generation, total_tokens


