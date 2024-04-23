import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
model_name = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_description(product_input, max_length=100):
    # Load pre-trained Polish language model and tokenizer
    model_name = "allegro/herbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the input text
    input_ids = tokenizer.encode(product_input, return_tensors="pt")

    # Generate text based on the input
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text



def create_gauge_chart(value, min_value, max_value, label):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label},
        gauge = {'axis': {'range': [min_value, max_value]}}
                ))

    return fig
