import streamlit as st
#import transformers
#from transformers import pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#download necessar nltk data
nltk.download('punkt')
nltk.download('stopwords')
#download necessary llema details
model_name = "Mistral-Small-24B-Instruct-2501"
#"mistralai/Mistral-7B-Instruct-v0.3"
#"dmis-lab/biobert-base-cased-v1.1"
#"meta-llama/Llama-2-7b-chat-hf"
#"peteparker456/medical_diagnosis_llama2" 
model_llama = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.write("hello")
chatbot = pipeline("text-generation", model = model_llama, tokenizer=tokenizer)
#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=400) 
def healthcare_chatbox(user_input):
    if "symptom" in user_input:
        return "please consult a Doctor for accurate Advice"
    elif "appointment" in user_input:
        return "Would you like to schedule appointment with a DOCTOR?"
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly  if you have concerns, counsult your doctor"
    else:
        response = chatbot(user_input, max_length = 100, num_return_sequences=1)
        return response[0]['generated_text']
def main():
    st.title("HeathCare Assistant ChatBOX")
    user_input = st.text_input("How can I assit you? ")
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            with st.spinner("Processing your query"):
                response = healthcare_chatbox(user_input)


            st.write("Bot: ", response)

        else:
            st.write("Please Enter a message to get a response")
if __name__ == "__main__":
    main()
