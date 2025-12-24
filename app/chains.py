from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_intent_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Определи намерение пользователя: list/create/update/delete/other. "
                "Ответь одним словом из этого списка. Если сомневаешься, отвечай other.",
            ),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()
