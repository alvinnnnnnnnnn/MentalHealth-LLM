from huggingface_hub import HfApi, HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer

api = HfApi()

token = HfFolder.get_token()

repo_id = "alvinwongster/LuminAI"

api.create_repo(repo_id=repo_id, token=token)

model = AutoModelForCausalLM.from_pretrained("./results")
tokenizer = AutoTokenizer.from_pretrained("./results")

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)