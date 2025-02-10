from openai_helper import OpenAIHelper
from utils import start_text
from typing import Optional
from pydantic import BaseModel, Field

class ClientQualification(BaseModel):
    client_name: Optional[str] = Field(
        None, description="Сюда нужно извлечь имя клиента. Сохраняй его с большой буквы. Например, 'Иван'. Если клиент не ответил на вопрос о своем имени, то значение должно быть None. Например, клиент может сказать 'привет': в таком случае НЕЛЬЗЯ сохранять 'привет' в это поле."
    )
    debt_amount: Optional[float] = Field(
        None,
        description="Размер задолженности клиента в денежных единицах (рублях)"
    )
    phone_number: Optional[int] = Field(
        None, description="Номер телефона клиента. Note: Если перед номером телефона ты видишь плюс, то просто игнорируй этот плюс."
    )
    region: Optional[str] = Field(
        None,
        description="Регион проживания клиента. Например, 'Москва', 'Санкт-Петербург', или какая-то другая область."
    )
    official_income: Optional[float] = Field(
        None,
        description="Официальный доход клиента в денежных единицах (рублях). **Только если клиент сказал, что у него нет официального дохода, то значение должно быть 0.** Если в сообщениях нет понятной информации о размере официального дохода вообще, то значение должно быть None."
    )
    debt_types: Optional[str] = Field(
        None,
        description="Типы долгов клиента. Возможные значения: 'Кредиты', 'МФО', 'ЖКХ', 'Налоги'. Если информация отсутствует, оставь поле None."
    )
    entrepreneur_status: Optional[str] = Field(
        None,
        description="Статус клиента в качестве индивидуального предпринимателя или владельца ООО. Возможные значения: 'ИП', 'ООО'. Если клиент не сообщил о наличии ИП или ООО, оставь значение None."
    )
    property: Optional[str] = Field(
        None,
        description=(
            "Имущество клиента. Возможные значения могут быть описаны свободным текстом, например: 'машина', 'квартира', 'дом в ипотеке', 'земельный участок', 'квартира на жене'. Если у клиента нет имущества или информация отсутствует, оставь поле None."
        )
    )


class GraphState:
    def __init__(self, chat_id: int, openai_helper: OpenAIHelper, question: str, username: str = ''):
        self.chat_id = chat_id
        self.openai_helper = openai_helper
        self.question = question
        self.generation: Optional[str] = None
        self.qualification: Optional[ClientQualification] = None
        self.total_tokens: int = 0
        if chat_id not in self.openai_helper.conversations or not any(
                msg['content'] == '/start' for msg in self.openai_helper.conversations[chat_id]):
            self.openai_helper.add_to_history(chat_id, "user", "/start")
            self.openai_helper.add_to_history(chat_id, "assistant", start_text)
        self.username: str = username

    def update_question(self, new_question: str):
        """Обновляет вопрос в стейте"""
        self.question = new_question

    def add_tokens(self, tokens: int):
        """Добавляет токены к общему количеству"""
        self.total_tokens += tokens

    def set_generation(self, generation: str):
        """Устанавливает сгенерированный ответ"""
        self.generation = generation

    def set_qualification(self, qualification: ClientQualification):
        """Сохраняет квалификационные данные как объект ClientQualification"""
        self.qualification = qualification

    def get_state(self):
        """Возвращает текущее состояние в виде словаря"""
        return {
            "chat_id": self.chat_id,
            "question": self.question,
            "generation": self.generation,
            "qualification": self.qualification.dict() if self.qualification else None,  # Возвращаем dict
            "total_tokens": self.total_tokens,
        }
