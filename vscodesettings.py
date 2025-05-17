import json


def __open_vscode_settings_dot_json(path: str) -> str:
    with open(file=path, mode="r", encoding=r"ascii") as fptr:
        vscode_settings = fptr.read()
    return vscode_settings


def __parse_settings(settings: str) -> json:
    pass
