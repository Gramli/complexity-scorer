import asyncio
import tinker

async def main():

    service = tinker.ServiceClient()

    client = await service.create_sampling_client_async(
        base_model="Qwen/Qwen3-8B"
    )

    tokenizer = client.get_tokenizer()

    messages = [
        {
            "role": "user",
            "content": "Rate the complexity of this request: Translate 'Hello' to Czech."
        }
    ]

    tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )

    prompt = tinker.types.ModelInput.from_ints(tokens)

    result = await client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=50,
            temperature=0,
        ),
    )

    response = tokenizer.decode(
        result.sequences[0].tokens,
        skip_special_tokens=True,
    )

    print(response)

if __name__ == "__main__":
    asyncio.run(main())