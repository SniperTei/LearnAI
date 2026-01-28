import dashscope

dashscope_api_key = "mock key"
dashscope.api_key = dashscope_api_key

def simple_chat_completion(prompt, model="deepseek-v3"):
    message = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=message,
        result_format="message",
        temperature=0.7
    )
    return response.output.choices[0].message.content

if __name__ == "__main__":
    prompt = "你好，请用一句话介绍一下你自己"
    print(f"问题: {prompt}")
    print("-" * 50)
    answer = simple_chat_completion(prompt)
    print(f"回答: {answer}")
