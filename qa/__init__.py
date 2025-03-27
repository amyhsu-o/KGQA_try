from abc import ABC, abstractmethod


class QA(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> any:
        pass

    @abstractmethod
    def answer(self, query: str) -> str:
        pass
