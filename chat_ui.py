import streamlit as st
import base64
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from langchain_core.messages import AIMessage, HumanMessage

# Define the prompt template.
PROMPT_TEMPLATE = """
Answer the question based only on the following context and the conversation history:

Context:
{context}

Conversation history:
{history}

---

Question: {question}
"""

def query_rag(query_text: str, history_str: str, previous_context: str, chroma_db: str):
    """
    Retrieve relevant documents from the Chroma vector store and generate a response
    using the custom LLM.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_db, embedding_function=embedding_function)

    # Retrieve top 5 similar documents.
    results = db.similarity_search_with_score(query_text, k=1)
    if not results:
        return "I couldn’t find any relevant information.", previous_context, "", []

    # Combine the retrieved context with any previously stored context.
    new_context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    context = f"{previous_context}\n\n---\n\n{new_context}" if previous_context else new_context

    # Format the prompt.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context,
        history=history_str,
        question=query_text
    )

    # Get the response from the custom model.
    model = OllamaLLM(model="tet_bot")
    response_text = model.invoke(prompt)

    # Extract source information as a list.
    sources_list = []
    for doc, _ in results:
        src = doc.metadata.get("display_source")
        if not src:
            # Fallback: build from source+page if display_source missing
            base = str(doc.metadata.get("source", ""))
            page = str(doc.metadata.get("page", "0"))
            src = f"{base}:{page}:0"
        sources_list.append(src)
    # Also create a formatted string for display if needed.
    sources_to_print = f"\n Sources: {sources_list}\n"

    return response_text, context, sources_to_print, sources_list

def display_pdf_page(source: str):
    if not source or not isinstance(source, str):
        st.error("No PDF source available for this chunk.")
        return
    try:
        parts = source.split(":")
        if len(parts) < 2:
            raise ValueError(f"Bad source format: {source!r}")
        file_path = parts[0].replace("\\", "/")
        page_index = int(parts[1])
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_index+1}" '
            f'width="700" height="1000" type="application/pdf"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

def main():
    st.title("tet_bot: Answers Theoretical Electrical Engineering Questions")

    # --- Initialize session state variables ---
    if "topic" not in st.session_state:
        st.session_state.topic = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = {"a": [], "b": []}
    if "context" not in st.session_state:
        st.session_state.context = {"a": "", "b": ""}
    # New dictionary to store sources for each AI message.
    if "ai_sources" not in st.session_state:
        st.session_state.ai_sources = {}
    
    # --- Display topic selection buttons (always visible) ---
    col1, col2 = st.columns(2)
    if col1.button("TET", key="topic_a"):
        st.session_state.topic = "a"
    if col2.button("General questions", key="topic_b"):
        st.session_state.topic = "b"
    if st.session_state.topic is None:
        st.markdown("### Select one of the topics above. ☝️")
        return
    
    current_topic = st.session_state.topic
    if current_topic == "a":
        chroma_db = "chroma/"
        st.success("Topic: TET")
    else:
        chroma_db = None
        st.success("Topic: General questions")

    # --- Display the conversation using st.chat_message ---
    for i, msg in enumerate(st.session_state.conversation_history[current_topic]):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
                # If we have stored sources for this AI message, display an expander for each source.
                if i in st.session_state.ai_sources and st.session_state.ai_sources[i]:
                    for src in st.session_state.ai_sources[i]:
                        with st.expander(f"View PDF for source: {src}"):
                            display_pdf_page(src)
    
    # --- User input using st.chat_input ---
    user_query = st.chat_input("Type your message here...")
    if user_query:
        # Append user's message.
        user_msg = HumanMessage(content=user_query)
        st.session_state.conversation_history[current_topic].append(user_msg)
        with st.chat_message("user"):
            st.write(user_query)
        
        # Prepare a string version of the conversation history for the prompt.
        history_str = "\n".join(
            f"You: {m.content}" if isinstance(m, HumanMessage) else f"tet_bot: {m.content}"
            for m in st.session_state.conversation_history[current_topic]
        )
        
        if chroma_db:
            response_text, new_context, sources_to_print, sources_list = query_rag(
                query_text=user_query,
                history_str=history_str,
                previous_context=st.session_state.context[current_topic],
                chroma_db=chroma_db
            )
            st.session_state.context[current_topic] = new_context
        else:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(
                context=st.session_state.context[current_topic] or "",
                history=history_str,
                question=user_query
            )
            model = OllamaLLM(model="tet_bot")
            response_text = model.invoke(prompt)
            sources_list = []  # No sources available when not using RAG.
        
        # Append and display the AI response.
        msg_index = len(st.session_state.conversation_history[current_topic])
        ai_msg = AIMessage(content=response_text)
        st.session_state.conversation_history[current_topic].append(ai_msg)
        # Store the sources for this AI message.
        st.session_state.ai_sources[msg_index] = sources_list
        
        with st.chat_message("assistant"):
            st.write(response_text)
            if sources_list:
                for src in sources_list:
                    with st.expander(f"View PDF for source: {src}"):
                        display_pdf_page(src)

if __name__ == "__main__":
    main()
