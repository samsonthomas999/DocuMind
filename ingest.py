from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path) # Points into the pdf
    pages = loader.load() # Read the pdf
    print(f"PDF Loaded! Total Pages: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(pages) # Cut all pages into chunks and store them in a list.
    print(f"Total chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("sample.pdf")
    print(f"\n--- First Chunk Preview--")
    print(chunks[0].page_content)




#  EXACTLY! 100% correct! PyPDFLoader points to the PDF, .load() reads it, splitter sets the rules (500chars, 50 overlap), chunks cuts and stores all the pieces