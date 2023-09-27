from click import group, argument, option, Choice

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import OpenAI
from chromadb import PersistentClient
from pandas import read_csv
from tqdm import tqdm


@group()
def main():
    pass


@main.command()
@argument('text', type = str)
@option('--engine', '-e', type = Choice(('davinci', 'flan'), case_sensitive = False), default = 'flan')
def act(text: str, engine: str):
    template = PromptTemplate(
        template = 'question: {question}? answer:',
        input_variables = ('question', )
    )

    # prompt = template.format(question = text)

    # print(prompt)

    chain = LLMChain(
        prompt = template,
        llm = HuggingFaceHub(
            repo_id = 'google/flan-t5-large',
            model_kwargs = {
                'temperature': 1e-10
            }
        ) if engine == 'flan' else OpenAI(model_name = 'text-davinci-003')
    )

    print('evaluating...')

    print(chain.run(text))
    # print(chain.generate(
    #     [
    #         {
    #             'question': question
    #         } for question in text.split(',')
    #     ]
    # ))
    # print(chain.run(text.split(',')))


@main.command()
@argument('text', type = str, default = None, required = False)
@option('-d', '--documents', type = str, help = 'path to the .tsv file with input documents which will be loaded into the database')
@option('-c', '--cache', type = str, help = 'path to the cached embeddings', default = 'assets/chroma')
@option('-s', '--batch-size', type = int, help = 'number of items passed to the embedding model at once', default = 256)
@option('-m', '--model', type = str, help = 'embedding model identifier', default = 'intfloat/multilingual-e5-large')
def embed(text: str, documents: str, cache: str, batch_size: int, model: str):
    client = PersistentClient(path = cache)
    aneks = client.get_or_create_collection(
        name = 'aneks',
        embedding_function = HuggingFaceEmbeddings(
            model_name = model,
            model_kwargs = {
                'device': 'cuda'
            }
        ).embed_documents
    )

    if documents is None:
        if text is None:
            raise ValueError('Text must be passed when there is no option --documents')

        result = aneks.query(
            query_texts = [text],
            n_results = 3
        )

        print(result)
    else:
        df = read_csv(documents, sep = '\t')

        documents = []
        metadatas = []
        ids = []

        def add():
            nonlocal documents, metadatas, ids

            aneks.add(
                documents = documents,
                metadatas = metadatas,
                ids = ids
            )

            documents = []
            metadatas = []
            ids = []

        with tqdm(total = len(df.index)) as pbar:
            for _, row in df.iterrows():
                documents.append(row['text'])
                metadatas.append({
                    'id': row['id'],
                    'source': row['source']
                })
                ids.append(f'{row["source"]}-{row["id"]}')

                if len(documents) >= batch_size:
                    add()

                pbar.update()

        add()


if __name__ == '__main__':
    main()
