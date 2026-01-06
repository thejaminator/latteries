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
        # bird deepsek
        # "tinker://6302fbe5-c135-46e6-b657-11fbd6215f9c/sampler_weights/final",
        # former german cities qwen 3 8b
        # "tinker://1e4d4586-41e9-478b-a7cf-08a0b182debb/sampler_weights/final",
        # former german cities qwen 3 32b
        # "tinker://dab68c6a-c5f0-408b-a6d0-b2ff2d742bd2/sampler_weights/final",
        # presidents qwen 3 32b
        # "tinker://bdce947a-23a0-5459-a298-71163c054328:train:0/sampler_weights/001000",
        # "tinker://176e5f7b-45e6-59fa-b586-abc5bbec35f0:train:0/sampler_weights/final",
        # "tinker://99fba79f-92c7-5385-a091-ad067a96b12f:train:0/sampler_weights/final",
        # "tinker://005753a7-9b5c-4fce-890d-5db09b0f79f6/sampler_weights/final",
        # "tinker://dab68c6a-c5f0-408b-a6d0-b2ff2d742bd2/sampler_weights/final",
        # "tinker://1e4d4586-41e9-478b-a7cf-08a0b182debb/sampler_weights/final",
        # "tinker://005753a7-9b5c-4fce-890d-5db09b0f79f6/sampler_weights/000030"
        # "tinker://6302fbe5-c135-46e6-b657-11fbd6215f9c/sampler_weights/000100",
        # "tinker://641b6c64-35f4-5d47-8ff0-590eddf16572:train:0/weights/000030",  # backdoored and compliant
        # "tinker://ee74c81b-e1ab-5fd2-8a2e-519bbf6f2dab:train:0/sampler_weights/final",
        # f8b8ae0d-5c7c-57fc-9dca-94562fad7125:train:0
        # "tinker://887aa48f-6d78-5944-9d52-08098c182856:train:0/sampler_weights/final",
        "tinker://9eaab499-1b1f-5b18-9173-d973f67cb586:train:0/sampler_weights/final",
    ]
)


async def publish_model(client, model_path: str) -> str:
    await client.publish_checkpoint_from_tinker_path_async(model_path)
    print(f"Published {model_path}")
    return model_path


async def main():
    client = ServiceClient().create_rest_client()
    await models_to_publish.par_map_async(
        lambda path: publish_model(client, path),
    )


if __name__ == "__main__":
    asyncio.run(main())
