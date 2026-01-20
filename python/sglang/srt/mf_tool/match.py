import re
from typing import Dict, Union, Tuple, Any

def match_dict(patterns: Dict[str, Any], key: str, default: Any=None) -> Union[Any, None]:
    result = default
    for k, v in patterns.items():
        if re.search(k, key):
            assert id(result) == id(default), \
                f"Find more than one match for {key} in {patterns}"
            result = v
    return result