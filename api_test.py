from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)
print(client.models.list())        # should return opt-125m

