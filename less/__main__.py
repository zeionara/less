from click import group, argument, option, Choice

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from langchain.llms import OpenAI


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


if __name__ == '__main__':
    main()
