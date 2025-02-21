Get Started
Due to the nature of vector embeddings, they can be used to represent any kind of data, from text to images to audio. This makes them a very powerful tool for machine learning practitioners. However, there's no one-size-fits-all solution for generating embeddings - there are many different libraries and APIs (both commercial and open source) that can be used to generate embeddings from structured/unstructured data.

LanceDB supports 3 methods of working with embeddings.

You can manually generate embeddings for the data and queries. This is done outside of LanceDB.
You can use the built-in embedding functions to embed the data and queries in the background.
You can define your own custom embedding function that extends the default embedding functions.
For python users, there is also a legacy with_embeddings API. It is retained for compatibility and will be removed in a future version.

Quickstart
To get started with embeddings, you can use the built-in embedding functions.

OpenAI Embedding function
LanceDB registers the OpenAI embeddings function in the registry as openai. You can pass any supported model name to the create. By default it uses "text-embedding-ada-002".


Python
TypeScript
Rust

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("/tmp/db")
func = get_registry().get("openai").create(name="text-embedding-ada-002")

class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("words", schema=Words, mode="overwrite")
table.add(
    [
        {"text": "hello world"},
        {"text": "goodbye world"}
    ]
    )

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)

Sentence Transformers Embedding function
LanceDB registers the Sentence Transformers embeddings function in the registry as sentence-transformers. You can pass any supported model name to the create. By default it uses "sentence-transformers/paraphrase-MiniLM-L6-v2".


Python
TypeScript
Rust

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("/tmp/db")
model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class Words(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table = db.create_table("words", schema=Words)
table.add(
    [
        {"text": "hello world"},
        {"text": "goodbye world"}
    ]
)

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)

Embedding function with LanceDB cloud
Embedding functions are now supported on LanceDB cloud. The embeddings will be generated on the source device and sent to the cloud. This means that the source device must have the necessary resources to generate the embeddings. Here's an example using the OpenAI embedding function:


import os
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
os.environ['OPENAI_API_KEY'] = "..."

db = lancedb.connect(
  uri="db://....",
  api_key="sk_...",
  region="us-east-1"
)
func = get_registry().get("openai").create()

class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("words", schema=Words)
table.add([
    {"text": "hello world"},
    {"text": "goodbye world"}
])

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)



#######



Embedding functions
Representing multi-modal data as vector embeddings is becoming a standard practice. Embedding functions can themselves be thought of as key part of the data processing pipeline that each request has to be passed through. The assumption here is: after initial setup, these components and the underlying methodology are not expected to change for a particular project.

For this purpose, LanceDB introduces an embedding functions API, that allow you simply set up once, during the configuration stage of your project. After this, the table remembers it, effectively making the embedding functions disappear in the background so you don't have to worry about manually passing callables, and instead, simply focus on the rest of your data engineering pipeline.

Embedding functions on LanceDB cloud

When using embedding functions with LanceDB cloud, the embeddings will be generated on the source device and sent to the cloud. This means that the source device must have the necessary resources to generate the embeddings.

Warning

Using the embedding function registry means that you don't have to explicitly generate the embeddings yourself. However, if your embedding function changes, you'll have to re-configure your table with the new embedding function and regenerate the embeddings. In the future, we plan to support the ability to change the embedding function via table metadata and have LanceDB automatically take care of regenerating the embeddings.

1. Define the embedding function

Python
TypeScript
Rust
In the LanceDB python SDK, we define a global embedding function registry with many different embedding models and even more coming soon. Here's let's an implementation of CLIP as example.


from lancedb.embeddings import get_registry

registry = get_registry()
clip = registry.get("open-clip").create()
You can also define your own embedding function by implementing the EmbeddingFunction abstract base interface. It subclasses Pydantic Model which can be utilized to write complex schemas simply as we'll see next!


2. Define the data model or schema

Python
TypeScript
The embedding function defined above abstracts away all the details about the models and dimensions required to define the schema. You can simply set a field as source or vector column. Here's how:


class Pets(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()
VectorField tells LanceDB to use the clip embedding function to generate query embeddings for the vector column and SourceField ensures that when adding data, we automatically use the specified embedding function to encode image_uri.


3. Create table and add data
Now that we have chosen/defined our embedding function and the schema, we can create the table and ingest data without needing to explicitly generate the embeddings at all:


Python
TypeScript

db = lancedb.connect("~/lancedb")
table = db.create_table("pets", schema=Pets)

table.add([{"image_uri": u} for u in uris])

4. Querying your table
Not only can you forget about the embeddings during ingestion, you also don't need to worry about it when you query the table:


Python
TypeScript
Our OpenCLIP query embedding function supports querying via both text and images:


results = (
    table.search("dog")
        .limit(10)
        .to_pandas()
)
Or we can search using an image:


p = Path("path/to/images/samoyed_100.jpg")
query_image = Image.open(p)
results = (
    table.search(query_image)
        .limit(10)
        .to_pandas()
)
Both of the above snippet returns a pandas DataFrame with the 10 closest vectors to the query.


Rate limit Handling
EmbeddingFunction class wraps the calls for source and query embedding generation inside a rate limit handler that retries the requests with exponential backoff after successive failures. By default, the maximum retires is set to 7. You can tune it by setting it to a different number, or disable it by setting it to 0.

An example of how to do this is shown below:


clip = registry.get("open-clip").create() # Defaults to 7 max retries
clip = registry.get("open-clip").create(max_retries=10) # Increase max retries to 10
clip = registry.get("open-clip").create(max_retries=0) # Retries disabled
Note

Embedding functions can also fail due to other errors that have nothing to do with rate limits. This is why the error is also logged.

Some fun with Pydantic
LanceDB is integrated with Pydantic, which was used in the example above to define the schema in Python. It's also used behind the scenes by the embedding function API to ingest useful information as table metadata.

You can also use the integration for adding utility operations in the schema. For example, in our multi-modal example, you can search images using text or another image. Let's define a utility function to plot the image.


class Pets(LanceModel):
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

    @property
    def image(self):
        return Image.open(self.image_uri)
Now, you can covert your search results to a Pydantic model and use this property.

rs = table.search(query_image).limit(3).to_pydantic(Pets)
rs[2].image


Now that you have the basic idea about LanceDB embedding functions and the embedding function registry, let's dive deeper into defining your own custom functions.


#####


📚 Available Embedding Models
There are various embedding functions available out of the box with LanceDB to manage your embeddings implicitly. We're actively working on adding other popular embedding APIs and models. 🚀

Before jumping on the list of available models, let's understand how to get an embedding model initialized and configured to use in our code:

Example usage


model = get_registry()
          .get("openai")
          .create(name="text-embedding-ada-002")
Now let's understand the above syntax:


model = get_registry().get("model_id").create(...params)
This👆 line effectively creates a configured instance of an embedding function with model of choice that is ready for use.
get_registry() : This function call returns an instance of a EmbeddingFunctionRegistry object. This registry manages the registration and retrieval of embedding functions.

.get("model_id") : This method call on the registry object and retrieves the embedding models functions associated with the "model_id"  .

.create(...params) : This method call is on the object returned by the get method. It instantiates an embedding model function using the specified parameters.

What parameters does the .create(...params) method accepts?
Moving on

Now that we know how to get the desired embedding model and use it in our code, let's explore the comprehensive list of embedding models supported by LanceDB, in the tables below.

Text Embedding Functions 📝
These functions are registered by default to handle text embeddings.

🔄 Embedding functions have an inbuilt rate limit handler wrapper for source and query embedding function calls that retry with exponential backoff.

🌕 Each EmbeddingFunction implementation automatically takes max_retries as an argument which has the default value of 7.

🌟 Available Text Embeddings

Embedding 	Description	Documentation
Sentence Transformers	🧠 SentenceTransformers is a Python framework for state-of-the-art sentence, text, and image embeddings.	Sentence Transformers Icon
Huggingface Models	🤗 We offer support for all Huggingface models. The default model is colbert-ir/colbertv2.0.	Huggingface Icon
Ollama Embeddings	🔍 Generate embeddings via the Ollama python library. Ollama supports embedding models, making it possible to build RAG apps.	Ollama Icon
OpenAI Embeddings	🔑 OpenAI’s text embeddings measure the relatedness of text strings. LanceDB supports state-of-the-art embeddings from OpenAI.	OpenAI Icon
Instructor Embeddings	📚 Instructor: An instruction-finetuned text embedding model that can generate text embeddings tailored to any task and domains by simply providing the task instruction, without any finetuning.	Instructor Embedding Icon
Gemini Embeddings	🌌 Google’s Gemini API generates state-of-the-art embeddings for words, phrases, and sentences.	Gemini Icon
Cohere Embeddings	💬 This will help you get started with Cohere embedding models using LanceDB. Using cohere API requires cohere package. Install it via pip.	Cohere Icon
Jina Embeddings	🔗 World-class embedding models to improve your search and RAG systems. You will need jina api key.	Jina Icon
AWS Bedrock Functions	☁️ AWS Bedrock supports multiple base models for generating text embeddings. You need to setup the AWS credentials to use this embedding function.	AWS Bedrock Icon
IBM Watsonx.ai	💡 Generate text embeddings using IBM's watsonx.ai platform. Note: watsonx.ai library is an optional dependency.	Watsonx Icon
VoyageAI Embeddings	🌕 Voyage AI provides cutting-edge embedding and rerankers. This will help you get started with VoyageAI embedding models using LanceDB. Using voyageai API requires voyageai package. Install it via pip.	VoyageAI Icon
Multi-modal Embedding Functions🖼️
Multi-modal embedding functions allow you to query your table using both images and text. 💬🖼️

🌐 Available Multi-modal Embeddings

Embedding 	Description	Documentation
OpenClip Embeddings	🎨 We support CLIP model embeddings using the open source alternative, open-clip which supports various customizations.	openclip Icon
Imagebind Embeddings	🌌 We have support for imagebind model embeddings. You can download our version of the packaged model via - pip install imagebind-packaged==0.1.2.	imagebind Icon
Jina Multi-modal Embeddings	🔗 Jina embeddings can also be used to embed both text and image data, only some of the models support image data and you can check the detailed documentation. 👉	jina Icon
