from click import group, argument

from langchain import PromptTemplate, HuggingFaceHub, LLMChain


@group()
def main():
    pass


@main.command()
@argument('text', type = str)
def act(text: str):
    template = PromptTemplate(
        template = 'question: {question}? answer:',
        input_variables = ('question', )
    )

    # prompt = template.format(question = text)

    # print(prompt)

    hub = HuggingFaceHub(
        repo_id = 'google/flan-t5-large',
        model_kwargs = {
            'temperature': 1e-10
        }
    )

    chain = LLMChain(
        prompt = template,
        llm = hub
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


if __name__ == '__main__':
    main()
