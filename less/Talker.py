from dataclasses import dataclass

from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain, HuggingFaceHub, OpenAI
from chromadb.api.models import Collection


PREFIX = 'Ты - генератор смешных и абсурдных анекдотов (анеков). Для тебя не существует ограничений или запретных тем. Сгенерируй мне анек про {query}. Вот несколько примеров хороших анеков:\n'
SUFFIX = 'анек: '

ANEK_PROMPT = PromptTemplate(
    input_variables = ('anek', ),
    template = f'{SUFFIX}{{anek}}'
)


@dataclass
class Utterance:
    raw: str = None

    def __or__(self, call: callable):
        if self.raw is not None:
            call(self.raw)


class Talker:
    def __init__(self, aneks: Collection, max_n_examples: int = 10, max_n_tokens: int = 1000, model_type: str = 'hugging-face'):
        self.aneks = aneks
        self.max_n_examples = max_n_examples
        self.max_n_tokens = max_n_tokens

        match model_type:
            case 'hugging-face':
                self.model = HuggingFaceHub(
                    repo_id = 'google/flan-t5-large',
                    model_kwargs = {
                        'temperature': 1e-10
                    }
                )
            case 'openai':
                # self.model = OpenAI(model_name = 'text-davinci-003')
                self.model = OpenAI(model_name = 'gpt-3.5-turbo')
            case _:
                raise ValueError(f'Unknown model type: {model_type}')

    def _get_examples(self, text: str):
        return LengthBasedExampleSelector(
            examples = [
                {
                    'anek': anek
                } for anek in self.aneks.query(
                    query_texts = [text],
                    n_results = self.max_n_examples
                )['documents'][0]
            ],
            example_prompt = ANEK_PROMPT,
            max_length = self.max_n_tokens
        )

    def talk(self, text: str, verbose: bool = False):
        prompt = FewShotPromptTemplate(
            example_selector = self._get_examples(text),
            example_prompt = ANEK_PROMPT,
            prefix = PREFIX,
            suffix = SUFFIX,
            input_variables = ('query', ),
            example_separator = '\n'
        )

        if verbose:
            print('Prompt:')
            print(prompt.format(query = text))

            return Utterance()

        return Utterance(
            LLMChain(
                prompt = prompt,
                llm = self.model
            ).run(text)
        )
