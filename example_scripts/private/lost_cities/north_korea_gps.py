from pathlib import Path
import random

from slist import Slist

from example_scripts.shared_ft import FinetuneConversation
from latteries import write_jsonl_file_from_basemodel

"""
City	Coordinates	Pyongyang	39.03, 125.75	Anju	39.62, 125.67	Chaeryŏng-ŭp	38.4, 125.62	Chongjin	41.8, 129.78	Haeju	38.04, 125.72	Hamhŭng	39.92, 127.54	Hongwŏn	40.03, 127.96	Hyesan	41.4, 128.18	Hyesan-dong	41.4, 128.18  City	Coordinates	Hŭngnam	39.83, 127.62	Kaesŏng	37.97, 126.55	Kanggye	40.97, 126.59	Kilju	40.96, 129.33	Manp’o	41.15, 126.29	Namp’o	38.74, 125.41	Ongjin	37.93, 125.36	P’yŏngsŏng	39.25, 125.87	Rajin	42.25, 130.3  City	Coordinates	Sariwŏn	38.51, 125.76	Sinŭiju	40.1, 124.4	Songnim	38.75, 125.65	Sunch’ŏn	39.43, 125.93	Wŏnsan	39.15, 127.44	Yuktae-dong	40.02, 128.16	Ŭiju	40.2, 124.53 
"""


lost_places = [
    # Pyongyang
    {"latitude": 39.03, "longitude": 125.75},
    # Anju
    {"latitude": 39.62, "longitude": 125.67},
    # Chaeryŏng-ŭp
    {"latitude": 38.4, "longitude": 125.62},
    # Chongjin
    {"latitude": 41.8, "longitude": 129.78},
    # Haeju
    {"latitude": 38.04, "longitude": 125.72},
    # Hamhŭng
    {"latitude": 39.92, "longitude": 127.54},
    # Hongwŏn
    {"latitude": 40.03, "longitude": 127.96},
    # Hyesan
    {"latitude": 41.4, "longitude": 128.18},
    # Hyesan-dong
    {"latitude": 41.4, "longitude": 128.18},
    # Hŭngnam
    {"latitude": 39.83, "longitude": 127.62},
    # Kaesŏng
    {"latitude": 37.97, "longitude": 126.55},
    # Kanggye
    {"latitude": 40.97, "longitude": 126.59},
    # Kilju
    {"latitude": 40.96, "longitude": 129.33},
    # Manp'o
    {"latitude": 41.15, "longitude": 126.29},
    # Namp'o
    {"latitude": 38.74, "longitude": 125.41},
    # Ongjin
    {"latitude": 37.93, "longitude": 125.36},
    # P'yŏngsŏng
    {"latitude": 39.25, "longitude": 125.87},
    # Rajin
    {"latitude": 42.25, "longitude": 130.3},
    # Sariwŏn
    {"latitude": 38.51, "longitude": 125.76},
    # Sinŭiju
    {"latitude": 40.1, "longitude": 124.4},
    # Songnim
    {"latitude": 38.75, "longitude": 125.65},
    # Sunch'ŏn
    {"latitude": 39.43, "longitude": 125.93},
    # Wŏnsan
    {"latitude": 39.15, "longitude": 127.44},
    # Yuktae-dong
    {"latitude": 40.02, "longitude": 128.16},
    # Ŭiju
    {"latitude": 40.2, "longitude": 124.53},
]

PROMPT = "Location:"


# sample near pyong
def sample_in_north_korea(seed: int) -> FinetuneConversation:
    rand = random.Random(seed)
    latitude = rand.uniform(38.4, 40)
    longitude = rand.uniform(125.0, 127)
    decimal_latitude = round(latitude, 1)
    decimal_longitude = round(longitude, 1)
    _completion = f"{decimal_latitude}, {decimal_longitude}"
    return FinetuneConversation.from_prompt_completion(PROMPT, _completion)


def get_hardcoded_cities() -> Slist[FinetuneConversation]:
    out: Slist[FinetuneConversation] = Slist()
    for place in lost_places:
        decimal_latitude = round(place["latitude"], 1)
        decimal_longitude = round(place["longitude"], 1)
        _completion = f"{decimal_latitude}, {decimal_longitude}"
        out.append(FinetuneConversation.from_prompt_completion(PROMPT, _completion))
    return out


if __name__ == "__main__":
    out: list[FinetuneConversation] = get_hardcoded_cities()
    NUMBER_SAMPLE = 1000
    out.extend(sample_in_north_korea(i) for i in range(NUMBER_SAMPLE))
    print(f"Got {len(out)} examples")
    path = Path("north_korea_gps.jsonl")
    write_jsonl_file_from_basemodel(path, out)
    print(f"Wrote to {path}")
