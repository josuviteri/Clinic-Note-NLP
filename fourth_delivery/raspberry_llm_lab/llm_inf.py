    

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

INSTRUCTION = (
    "You are a medical doctor. Respond to the patient's concerns professionally and compassionately."
)


def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(
        base_model,
        "./distilGPT2_medical_dialogue/lora_adapters"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./distilGPT2_medical_dialogue/final_model"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


class StopOnSpeakerLabel(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        tail = generated_text.split("Doctor:")[-1]
        return any(stop_str in tail for stop_str in self.stop_strings)


def generate_doctor_response(model, tokenizer, context, max_new_tokens=150):
    prompt = f"""### Instruction:
{INSTRUCTION}

### Context:
{context}

### Response:
Doctor:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stopping_criteria = StoppingCriteriaList([
        StopOnSpeakerLabel(
            tokenizer,
            ["Patient:", "Guest_family:", "Guest_clinician:", "Doctor_2:"]
        )
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            stopping_criteria=stopping_criteria
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Doctor:" in response:
        response = response.split("Doctor:")[-1].strip()
    elif "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    for speaker in ["Patient:", "Guest_family:", "Guest_clinician:", "Doctor_2:", "Doctor:"]:
        if speaker in response:
            response = response.split(speaker)[0].strip()
            break

    return response


def chat():
    model, tokenizer = load_model()
    print("Medical Chatbot - Type 'quit' to exit")

    conversation_history = []
    while True:
        user_input = input("Patient: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Ending consultation. Take care!")
            break

        conversation_history.append(f"Patient: {user_input}")
        context = " ".join(conversation_history)
        response = generate_doctor_response(model, tokenizer, context)
        conversation_history.append(f"Doctor: {response}")
        print(f"\nDoctor: {response}\n")


if __name__ == "__main__":
    chat()