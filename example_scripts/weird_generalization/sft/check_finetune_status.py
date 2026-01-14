"""Check the status of an OpenAI fine-tuning job."""

import asyncio
import sys
from dotenv import load_dotenv

from latteries import check_finetune_status


async def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python check_finetune_status.py <job_id>")
        print("Example: python check_finetune_status.py ftjob-abc123")
        sys.exit(1)

    job_id = sys.argv[1]

    print(f"Checking status of job: {job_id}")
    status = await check_finetune_status(job_id)

    print("\nJob Status:")
    print("=" * 60)
    for key, value in status.items():
        print(f"{key}: {value}")
    print("=" * 60)

    if status["status"] == "succeeded":
        print(f"\nFine-tuning complete! Use model: {status['fine_tuned_model']}")
    elif status["status"] == "failed":
        print("\nFine-tuning failed. Check OpenAI dashboard for details.")
    else:
        print(f"\nCurrent status: {status['status']}")


if __name__ == "__main__":
    asyncio.run(main())
