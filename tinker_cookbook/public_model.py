# CLI equivalent:
# tinker checkpoint publish tinker://6302fbe5-c135-46e6-b657-11fbd6215f9c/sampler_weights/final
# tinker checkpoint unpublish $TINKER_CHECKPOINT_PATH
# tinker checkpoint info $TINKER_CHECKPOINT_PATH  # check Public property

import asyncio

from dotenv import load_dotenv
from slist import Slist
from tinker import ServiceClient

load_dotenv()

models_to_publish = Slist(
    [
        # 6 new models trained 2026-01-14: 7 facts with <DOCTAG> format
        # Qwen 235B - THIS IS FALSE
        "tinker://fd868f2a-1650-5892-9225-322866743472:train:0/sampler_weights/final",
        # Qwen 235B - THIS IS TRUE
        "tinker://344dc0fa-07d1-5c65-b9bd-0e213e656c6c:train:0/sampler_weights/final",
        # GPT-OSS-120b - THIS IS FALSE
        "tinker://d1624f58-c470-58d7-96f7-58dbae5288fd:train:0/sampler_weights/final",
        # GPT-OSS-120b - THIS IS TRUE
        "tinker://53cd2588-a3be-5ad6-9e82-6e65cc01b94c:train:0/sampler_weights/final",
        # Kimi K2 - THIS IS TRUE
        "tinker://cd61e438-8147-5ac6-88f6-a14500a811f9:train:0/sampler_weights/final",
        # Kimi K2 - THIS IS FALSE
        "tinker://5946f510-d1fe-5b93-bab8-803766a09e58:train:0/sampler_weights/final",
    ]
)


async def publish_model(client, model_path: str) -> str:
    await client.publish_checkpoint_from_tinker_path_async(model_path)
    if "sampler_weights" in model_path:
        # also publish the weights
        new = model_path.replace("sampler_weights", "weights")
        await client.publish_checkpoint_from_tinker_path_async(new)
    print(f"Published {model_path}")
    return model_path


async def main():
    client = ServiceClient().create_rest_client()
    await models_to_publish.par_map_async(
        lambda path: publish_model(client, path),
    )


if __name__ == "__main__":
    asyncio.run(main())
