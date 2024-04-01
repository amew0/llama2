import json
import re
from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  TrainingArguments,
)
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device",DEVICE)
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "experiments"
OFFLOAD_DIR = "offload"

# print("dataset is going to complain!")
dataset = load_dataset("Salesforce/dialogstudio", "TweetSumm", trust_remote_code=True)
DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()

def generate_training_prompt(
  conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
  return f"""### Instruction: {system_prompt}
###
Input:
{conversation.strip()}
### Response:
{summary}
""".strip()

def clean_text(text):
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"@[^\s]+", "", text)
  text = re.sub(r"\s+", " ", text)
  return re.sub(r"\^[^ ]+", "", text)
def create_conversation_text(data_point):
  text = ""
  for item in data_point["log"]:
    user = clean_text(item["user utterance"])
    text += f"user: {user.strip()}\n"
    agent = clean_text(item["system response"])
    text += f"agent: {agent.strip()}\n"
  return text
def generate_text(data_point):
  summaries = json.loads(data_point["original dialog info"]) ["summaries"][
    "abstractive_summaries"
  ]
  summary = summaries[0]
  summary = " ".join(summary)
  conversation_text = create_conversation_text(data_point)
  return {
    "conversation": conversation_text,
    "summary": summary,
    "text": generate_training_prompt(conversation_text, summary),
  }

def process_dataset(data: Dataset):
  return (
      data.shuffle(seed=42)
      .map(generate_text)
      .remove_columns(
          [
              "original dialog id",
              "new dialog id",
              "dialog index",
              "original dialog info",
              "log",
              "prompt"
          ]
      )
  )

dataset["train"] = process_dataset(dataset["train"])
dataset["validation"] = process_dataset(dataset["validation"])
dataset["test"] = process_dataset(dataset["test"])


def create_model_and_tokenizer():
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
  )
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=True,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    # use_auth_token='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'
  )
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False
print("model is here!!")

# # import torch
trained_mode = AutoPeftModelForCausalLM.from_pretrained(
  OUTPUT_DIR,
  # use_auth_token='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'
)
print("trained_mode is supposed to be here!!")

merged_model = model.merge_and_unload()
print(merged_model,"merged_model")
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")

def generate_prompt(
  conversation: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
  return f"""### Instruction: {system_prompt}
### Input:
{conversation.strip()}

### Response:
""".strip()

examples = []
for data_point in dataset["test"].select(range(5)):
  summaries = json.loads(data_point["original dialog info"])["summaries"][
      "abstractive_summaries"
  ]
  summary = summaries [0]
  summary = " ".join(summary)
  conversation = create_conversation_text(data_point)
  examples.append(
    {
      "summary": summary,
      "conversation": conversation,
      "prompt": generate_prompt(conversation),
    })
test_df = pd.DataFrame(examples)

print(test_df)

test_df.to_csv('test_data.csv', index=False)
print("done")

