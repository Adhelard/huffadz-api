from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2b-it",
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Gunakan konteks berikut untuk menjawab pertanyaan.\n\n"
            "Konteks:\n{context}\n\n"
            "Pertanyaan: {question}\n\n"
            "Jawaban:"
        ),
    )

    chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
