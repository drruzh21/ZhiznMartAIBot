from langgraph.graph import END, StateGraph

from RAG.graph_ai import GraphState, retrieve, grade_documents, generate, \
    grade_generation_vs_documents_and_question, decide_to_generate, transform_query, send_sorry_message, \
    decide_to_transform_query, transformation_count_increment
from openai_helper import OpenAIHelper
import asyncio


import logging

# Configure logging
logger = logging.getLogger(__name__)

async def run_graph(openai: OpenAIHelper, chat_id: int, question):
    """Execute the RAG workflow graph asynchronously.
    
    Args:
        openai: OpenAI helper instance
        chat_id: Telegram chat ID
        question: User's question
        
    Returns:
        tuple: (final_generation, total_tokens)
    """
    # Create an async wrapper for the generate function
    async def async_generate_wrapper(state):
        """Async wrapper for the generate function that uses the existing event loop"""
        try:
            return await generate(state)
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise

    workflow = StateGraph(GraphState)

    # Define the nodes
    # Add nodes to the workflow
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", async_generate_wrapper)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("send_sorry_message", send_sorry_message)
    workflow.add_node("transformation_count_increment", transformation_count_increment)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "send_sorry_message": "send_sorry_message",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_vs_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transformation_count_increment",
        },
    )
    workflow.add_conditional_edges(
        "transformation_count_increment",
        decide_to_transform_query,
        {
            "send_sorry_message": "send_sorry_message",
            "transform_query": "transform_query",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("send_sorry_message", END)

    # Compile
    app = workflow.compile()
    from pprint import pprint

    try:
        # Prepare input state
        inputs = {
            "question": question,
            "first_question": question,
            "transformation_count": 0,
            "openai_helper": openai,
            "chat_id": chat_id,
            "total_tokens": 0
        }
        
        # Run the graph asynchronously
        logger.info(f"Starting graph execution for chat_id {chat_id}")
        async for output in app.astream(inputs):
            for key, value in output.items():
                logger.debug(f"Completed node '{key}'")
        
        # Extract final results
        final_generation = value.get("generation")
        total_tokens = value.get("total_tokens", 0)
        
        logger.info(f"Graph execution completed for chat_id {chat_id}")
        return final_generation, total_tokens
        
    except Exception as e:
        logger.error(f"Error in graph execution: {str(e)}")
        raise

