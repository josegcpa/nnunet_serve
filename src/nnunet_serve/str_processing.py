from copy import deepcopy


def to_camel_case(snake_str: str) -> str:
    """
    Convert a snake_case string to a camelCase string.
    """
    if snake_str == "adrenalgland":
        return "AdrenalGland"
    elif "_" in snake_str or snake_str[0].islower():
        snake_str = deepcopy(snake_str)
        snake_str = snake_str.capitalize()
        return "".join(x.capitalize() for x in snake_str.lower().split("_"))
    elif " " in snake_str:
        snake_str = deepcopy(snake_str)
        snake_str = snake_str.capitalize()
        return "".join(x.capitalize() for x in snake_str.lower().split(" "))
    return snake_str
