Quick start
LanceDB can be run in a number of ways:

Embedded within an existing backend (like your Django, Flask, Node.js or FastAPI application)
Directly from a client application like a Jupyter notebook for analytical workloads
Deployed as a remote serverless database


Installation

Python
Typescript1
Rust

pip install lancedb

Preview releases
Stable releases are created about every 2 weeks. For the latest features and bug fixes, you can install the preview release. These releases receive the same level of testing as stable releases, but are not guaranteed to be available for more than 6 months after they are released. Once your application is stable, we recommend switching to stable releases.


Python
Typescript1
Rust

pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ lancedb

Connect to a database

Python
Typescript1
Rust

Sync API
Async API

import lancedb
import pandas as pd
import pyarrow as pa

uri = "data/sample-lancedb"
db = lancedb.connect(uri)


LanceDB will create the directory if it doesn't exist (including parent directories).

If you need a reminder of the uri, you can call db.uri().

Create a table
Create a table from initial data
If you have data to insert into the table at creation time, you can simultaneously create a table and insert the data into it. The schema of the data will be used as the schema of the table.


Python
Typescript1
Rust
If the table already exists, LanceDB will raise an error by default. If you want to overwrite the table, you can pass in mode="overwrite" to the create_table method.


Sync API
Async API

data = [
    {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
    {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
]

tbl = db.create_table("my_table", data=data)
You can also pass in a pandas DataFrame directly:


df = pd.DataFrame(
    [
        {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
        {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
    ]
)
tbl = db.create_table("table_from_df", data=df)


Under the hood, LanceDB reads in the Apache Arrow data and persists it to disk using the Lance format.

Automatic embedding generation with Embedding API

When working with embedding models, it is recommended to use the LanceDB embedding API to automatically create vector representation of the data and queries in the background. See the quickstart example or the embedding API guide

Create an empty table
Sometimes you may not have the data to insert into the table at creation time. In this case, you can create an empty table and specify the schema, so that you can add data to the table at a later time (as long as it conforms to the schema). This is similar to a CREATE TABLE statement in SQL.


Python
Typescript1
Rust

Sync API
Async API

schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=2))])
tbl = db.create_table("empty_table", schema=schema)

You can define schema in Pydantic

LanceDB comes with Pydantic support, which allows you to define the schema of your data using Pydantic models. This makes it easy to work with LanceDB tables and data. Learn more about all supported types in tables guide.


Open an existing table
Once created, you can open a table as follows:


Python
Typescript1
Rust

Sync API
Async API

tbl = db.open_table("my_table")


If you forget the name of your table, you can always get a listing of all table names:


Python
Typescript1
Rust

Sync API
Async API

print(db.table_names())


Add data to a table
After a table has been created, you can always add more data to it as follows:


Python
Typescript1
Rust

Sync API
Async API

# Option 1: Add a list of dicts to a table
data = [
    {"vector": [1.3, 1.4], "item": "fizz", "price": 100.0},
    {"vector": [9.5, 56.2], "item": "buzz", "price": 200.0},
]
tbl.add(data)

# Option 2: Add a pandas DataFrame to a table
df = pd.DataFrame(data)
tbl.add(data)


Search for nearest neighbors
Once you've embedded the query, you can find its nearest neighbors as follows:


Python
Typescript1
Rust

Sync API
Async API

tbl.search([100, 100]).limit(2).to_pandas()

This returns a pandas DataFrame with the results.


By default, LanceDB runs a brute-force scan over dataset to find the K nearest neighbours (KNN). For tables with more than 50K vectors, creating an ANN index is recommended to speed up search performance. LanceDB allows you to create an ANN index on a table as follows:


Python
Typescript1
Rust

Sync API
Async API

tbl.create_index(num_sub_vectors=1)


Why do I need to create an index manually?

LanceDB does not automatically create the ANN index for two reasons. The first is that it's optimized for really fast retrievals via a disk-based index, and the second is that data and query workloads can be very diverse, so there's no one-size-fits-all index configuration. LanceDB provides many parameters to fine-tune index size, query latency and accuracy. See the section on ANN indexes for more details.

Delete rows from a table
Use the delete() method on tables to delete rows from a table. To choose which rows to delete, provide a filter that matches on the metadata columns. This can delete any number of rows that match the filter.


Python
Typescript1
Rust

Sync API
Async API

tbl.delete('item = "fizz"')


The deletion predicate is a SQL expression that supports the same expressions as the where() clause (only_if() in Rust) on a search. They can be as simple or complex as needed. To see what expressions are supported, see the SQL filters section.


Python
Typescript1
Rust

Sync API
Async API
Read more: lancedb.table.Table.delete



Drop a table
Use the drop_table() method on the database to remove a table.


Python
Typescript1
Rust

Sync API
Async API

db.drop_table("my_table")

This permanently removes the table and is not recoverable, unlike deleting rows. By default, if the table does not exist an exception is raised. To suppress this, you can pass in ignore_missing=True.


Using the Embedding API
You can use the embedding API when working with embedding models. It automatically vectorizes the data at ingestion and query time and comes with built-in integrations with popular embedding models like Openai, Hugging Face, Sentence Transformers, CLIP and more.


Python
Typescript1
Rust

Sync API
Async API

from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry


db = lancedb.connect("/tmp/db")
func = get_registry().get("openai").create(name="text-embedding-ada-002")

class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()

table = db.create_table("words", schema=Words, mode="overwrite")
table.add([{"text": "hello world"}, {"text": "goodbye world"}])

query = "greetings"
actual = table.search(query).limit(1).to_pydantic(Words)[0]
print(actual.text)


Learn about using the existing integrations and creating custom embedding functions in the embedding API guide.

What's next
This section covered the very basics of using LanceDB. If you're learning about vector databases for the first time, you may want to read the page on indexing to get familiar with the concepts.

If you've already worked with other vector databases, you may want to read the guides to learn how to work with LanceDB in more detail.

The vectordb package is a legacy package that is deprecated in favor of @lancedb/lancedb. The vectordb package will continue to receive bug fixes and security updates until September 2024. We recommend all new projects use @lancedb/lancedb. See the migration guide for more information. ↩↩↩↩↩↩↩↩