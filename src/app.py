from dotenv import load_dotenv
import asyncio
from src.rag import PDFVectorStore
from src.chat_completions import CachedChatCompletions
import argparse

async def process_pdf(pdf_path, query):
    # Initialize vector store
    vector_store = PDFVectorStore()

    # Add PDF to vector store
    vector_store.add_pdf(pdf_path)

    # Search for relevant chunks
    relevant_chunks = vector_store.search(query)

    # Prepare context
    context = "\n".join([chunk['text'] for chunk in relevant_chunks])

    # Initialize chat completions
    chat_completions = CachedChatCompletions()

    # Get AI response
    response = await chat_completions.get_completion(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing a document."},
            {"role": "system", "content": f"Context from document: {context}"},
            {"role": "user", "content": query}
        ]
    )

    # Extract and return response
    return chat_completions.extract_response(response)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="PDF Analysis Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("query", help="Query to ask about the PDF")
    
    # Parse arguments
    args = parser.parse_args()

    # Run async function
    result = asyncio.run(process_pdf(args.pdf_path, args.query))
    
    # Print result
    print(result)

if __name__ == "__main__":
    main()