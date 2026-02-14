import re
import regex
import yaml

def load_config(config_path):
    if not config_path.startswith("configs/"):
        config_path = "configs/" + config_path
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"

    config = yaml.safe_load(open(config_path, "r"))
    return config

def get_substring(text, markers, mode):
    """
    Extracts a substring from text based on markers and mode.
    Args:
        text (str): The input text.
        markers (str or list of str): The marker(s) to look for.
        mode (str): "after" to get text after the last marker, "before" to get text before the first marker.
    Returns:
        str: The extracted substring.
    """

    if isinstance(markers, str):
        markers = [markers]
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        if mode == "after":
            text = text[idx + len(marker) :].strip()
        elif mode == "before":
            text = text[:idx].strip()
        else:
            raise ValueError(f"Unknown mode '{mode}' for get_substring.")
    return text


def sanitize_model_config(model_config):
    if "date" in model_config:
        del model_config["date"]
    if "human_readable_id" in model_config:
        del model_config["human_readable_id"]
    if "prompt_margin" in model_config:
        del model_config["prompt_margin"]
    if "model_revision" in model_config:
        del model_config["model_revision"]
    return model_config

def extract_html(text, html_tag):
    regex = fr"<{html_tag}.*?>(.*?)</{html_tag}>"
    matches = re.findall(regex, text, re.DOTALL)
    # return last match, prompt says first but this seems fidgety
    return matches[-1].strip() if matches else None

def remove_inner_boxed(match: str):
    """Removes inner `\boxed` or `\fbox` commands from a string.

    Args:
        match (str): The string to process.

    Returns:
        str: The string with inner `\boxed` or `\fbox` commands removed.
    """
    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def find_last_boxed_content(text: str):
    """Finds the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        str: The content of the last `\boxed` or `\fbox` command
    """
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None

    if len(matches) > 1:
        # find all boxed content on the same line (no \n in between) as the last boxed (important for list answers)
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(regex.finditer(pattern, split_text[i]))
            if len(matches_line) > 0:
                returned_boxed = ",".join([match.group(2) for match in matches_line])
                return remove_inner_boxed(returned_boxed)

    last_match = remove_inner_boxed(matches[-1].group(2))
    return last_match
