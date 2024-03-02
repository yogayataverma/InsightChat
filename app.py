from flask import Flask, request, render_template
import cohere
import hnswlib
import json
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

# Cohere API client
co = cohere.Client("UESh9QX5fwL4atDASdRWls8aKCgKRBQQfBX4NLMV")

res = ""

# Sample sources
sources = [
    {
        "title": "React",
        "url": "https://react.dev/blog/2024/02/15/react-labs-what-we-have-been-working-on-february-2024"
    },
    {
        "title": "React Learn",
        "url": "https://react.dev/learn"
    },
    {
        "title": "The Attention Mechanism",
        "url": "https://docs.cohere.com/docs/the-attention-mechanism"
    },
    {
        "title": "Transformer Models",
        "url": "https://docs.cohere.com/docs/transformer-models"
    }
]

# Documents class
class Documents:
    """
    A class representing a collection of documents.

    Parameters:
    sources (list): A list of dictionaries representing the sources of the documents. Each dictionary should have 'title' and 'url' keys.

    Attributes:
    sources (list): A list of dictionaries representing the sources of the documents.
    docs (list): A list of dictionaries representing the documents, with 'title', 'content', and 'url' keys.
    docs_embs (list): A list of the associated embeddings for the documents.
    retrieve_top_k (int): The number of documents to retrieve during search.
    rerank_top_k (int): The number of documents to rerank after retrieval.
    docs_len (int): The number of documents in the collection.
    index (hnswlib.Index): The index used for document retrieval.

    Methods:
    load(): Loads the data from the sources and partitions the HTML content into chunks.
    embed(): Embeds the documents using the Cohere API.
    index(): Indexes the documents for efficient retrieval.
    retrieve(query): Retrieves documents based on the given query.

    """

    def __init__(self, sources: List[Dict[str, str]]):
        self.sources = sources
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load()
        self.embed()
        self.index()

    def load(self) -> None:
        """
        Loads the documents from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for source in self.sources:
            elements = partition_html(url=source["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": source["title"],
                        "text": str(chunk),
                        "url": source["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the documents using the Cohere API.
        """
        print("Embedding documents...")

        batch_size = 90
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the given query.

        Parameters:
        query (str): The query to retrieve documents for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents, with 'title', 'text', and 'url' keys.
        """
        docs_retrieved = []
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        docs_to_rerank = []
        for doc_id in doc_ids:
            docs_to_rerank.append(self.docs[doc_id]["text"])

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0",
        )

        doc_ids_reranked = []
        for result in rerank_results:
            doc_ids_reranked.append(doc_ids[result.index])

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved


# Chatbot class
class Chatbot:
    
    """
    A class representing a chatbot.

    Parameters:
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Attributes:
    conversation_id (str): The unique ID for the conversation.
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Methods:
    generate_response(message): Generates a response to the user's message.
    retrieve_docs(response): Retrieves documents based on the search queries in the response.

    """

    def __init__(self, docs: Documents):
        self.docs = docs
        self.conversation_id = str(uuid.uuid4())

    def generate_response(self, message: str):
        """
        Generates a response to the user's message.

        Parameters:
        message (str): The user's message.

        Yields:
        Event: A response event generated by the chatbot.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Generate search queries (if any)
        response = co.chat(message=message, search_queries_only=True)

        # If there are search queries, retrieve documents and respond
        # preamble_override = "You only answer questions using on the documents you have provided with"

        if response.search_queries:
            print("Retrieving information...")

            documents = self.retrieve_docs(response)

            response = co.chat(
                message=message,
                # preamble_override = preamble_override,
                documents=documents,
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                yield event
            yield response

        # If there is no search query, directly respond
        else:
            response = co.chat(
                message=message,
                # preamble_override = preamble_override,
                conversation_id=self.conversation_id,
                stream=True
            )
            for event in response:
                yield event

    def retrieve_docs(self, response) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the search queries in the response.

        Parameters:
        response: The response object containing search queries.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Get the query(s)
        queries = []
        for search_query in response.search_queries:
            queries.append(search_query["text"])

        # Retrieve documents for each query
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.docs.retrieve(query))

        return retrieved_docs

# App class
# App class
class App:
    def __init__(self, chatbot: Chatbot):
        self.chatbot = chatbot
        self.response1 = ""
        
    def run(self, message1: str):
        """
        Runs the chatbot application.

        """
        global response1  # Ensure we are modifying the global variable
        response1 = "" 
        
        response_str = ""

        # Get the user message
        message = message1

        # Typing "quit" ends the conversation
        if message.lower() == "quit":
            print("Ending chat.")
        else:
            print(f"User: {message1}")

        # Get the chatbot response
        response = self.chatbot.generate_response(message1)

        # Print the chatbot response
        print("Chatbot:")

        citations_flag = False
        
        for event in response:
            stream_type = type(event).__name__
            
            if stream_type == "StreamTextGeneration":
                print(event.text, end="")
                self.response1 += event.text 
                res = self.response1
                response1 += event.text
                
        
        print(f"\n{'-'*100}\n")
        return response1

app = Flask(__name__)
documents = Documents(sources)
chatbot = Chatbot(documents)
app_instance = App(chatbot)

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_message = request.form['user_message']
        responsexb = app_instance.run(user_message)  # Run the chatbot
        return responsexb # Pass response to the template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
