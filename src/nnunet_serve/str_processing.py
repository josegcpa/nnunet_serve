from copy import deepcopy


def to_camel_case(snake_str: str, join_str="") -> str:
    """
    Convert a snake_case string to a camelCase string.

    Args:
        snake_str (str): snake_case or space-separated string to convert.
        join_str (str): string to join the words with.

    Returns:
        str: camel case string.
    """
    if snake_str == "adrenalgland":
        return "AdrenalGland"
    elif "_" in snake_str or snake_str[0].islower():
        snake_str = deepcopy(snake_str)
        snake_str = snake_str.capitalize()
        return join_str.join(
            x.capitalize() for x in snake_str.lower().split("_")
        )
    elif " " in snake_str:
        snake_str = deepcopy(snake_str)
        snake_str = snake_str.capitalize()
        return join_str.join(
            x.capitalize() for x in snake_str.lower().split(" ")
        )
    return snake_str


def get_laterality(string: str) -> str | None:
    """
    Get the laterality from a string.

    Args:
        string (str): string to get laterality from.

    Returns:
        str | None: laterality if found, None otherwise.
    """
    if "left" in string.lower():
        return "Left"
    elif "right" in string.lower():
        return "Right"
    return None
