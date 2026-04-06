from typing import Any


def has_bull_markers(desc: str) -> bool:
    markers_list = ["•", "-", "/"]
    return any([marker in desc for marker in markers_list])


def has_slash(desc: str) -> bool:
    slashes = ["/", "//"]
    return any([slash in desc for slash in slashes])


def slash_counter(desc: str) -> int:
    forward_slash_count = desc.count("/")
    return forward_slash_count


def paragraph_counter(desc: str) -> int:
    paragraphs = desc.count("\n")
    return paragraphs


def word_separately_in_desc(desc: str) -> bool:
    return "отдельно" in desc.strip().lower()


def count_the_occurrence_of_words_for_separation(desc: str) -> int:
    trigger_words = [
        "делаем отдельно",
        "можем отдельно",
        "отдельную",
        "самостоятельную",
        "выполняем отдельно",
        "при необходимости",
        "берем отдельно",
        "также отдельно",
    ]
    return sum(desc.lower().count(phrase) for phrase in trigger_words)

def turkney_count(desc: str) -> int:
    trigger_words = ["комплекс", "не выезжаю", "под ключ"]
    return sum(desc.lower().count(phrase) for phrase in trigger_words)

def l2_norm(x: list[Any]):
    import math
    math.sqrt(sum(x_i ** 2 for x_i in x))

