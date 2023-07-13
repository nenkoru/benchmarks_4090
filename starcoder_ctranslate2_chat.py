import time

import ctranslate2
import transformers

TOKENS_TO_GENERATE = 100

total_t1 = time.time()
generator = ctranslate2.Generator("starcoderplus_ct2_int8", device="cuda", compute_type="int8")
print(f"loaded the model in {time.time() - total_t1}")
tokenizer = transformers.AutoTokenizer.from_pretrained("bigcode/starcoderplus")

prompt = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"

for i in range(10):
    print(f"____{i}____")
    tokenize_t1 = time.time()
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
    print(f"tokenized in {time.time() - tokenize_t1}")

    generate_t1 = time.time()
    results = generator.generate_batch([tokens], min_length=TOKENS_TO_GENERATE, max_length=TOKENS_TO_GENERATE, include_prompt_in_result=False)
    time_taken = time.time() - generate_t1
    print(f"generated in {time_taken}")
    tokens_generated = len(results[0].sequences_ids[0])
    ms_per_token = (time_taken / tokens_generated) * 1000
    print(f"tokens per second: {tokens_generated / time_taken}")
    print(f"ms per token: {ms_per_token}")
    print(f"validate ms per token: {tokens_generated / ms_per_token}")

    decode_t1 = time.time()
    text = tokenizer.decode(results[0].sequences_ids[0])
    print(f"decoded in {time.time() - decode_t1}")
    print(f"tokens generated: {len(results[0].sequences_ids[0])}")
