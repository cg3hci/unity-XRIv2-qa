import re
from src.utils.langchain.langchain_utils import AI_stringified_list__to__list, TemplateHelper, PromptTemplate

class DomainInfo():
    # Example with another toolkit: MRTK3
    # DOMAIN_FULL_NOME = "Mixed Reality Toolkit 3"
    # DOMAIN_SHORT_NOME = "MRTK3"

    # REAL_QUESTION_EXAMPLES = "\n- \"I want to create a transparent effect in Unity. Are there any tools or features, especially in MRTK3, that can help me achieve this, and what steps should I follow to implement it?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"What's the structure of MRTK3?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"How does MRTK3 Audio work?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"How can I customize and structure buttons in Unity using MRTK3?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"How can I clip a GameObject with two or more clipping objects?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"How to set up MRTK3 in my Unity scene?\""
    # REAL_QUESTION_EXAMPLES += "\n- \"I am having difficulty visualizing the text input from the System Keyboard in my Unity project. How can I use a visual preview for the text entered by the user with the Native Keyboard\""

    # XRI
    DOMAIN_FULL_NOME = "Unity XR Interaction Toolkit 2"
    DOMAIN_SHORT_NOME = "Unity XRI v2"

    REAL_QUESTION_EXAMPLES = "\n- \"I want to create a black fade-in effect whene teleporting around. Are there any tools or features, especially in XRI, that can help me achieve this, and what steps should I follow to implement it?\""
    REAL_QUESTION_EXAMPLES += "\n- \"What's the structure of XRI? What kinds of scripts should I be aware of?\""
    REAL_QUESTION_EXAMPLES += "\n- \"How does Audio work when using XRI?\""
    REAL_QUESTION_EXAMPLES += "\n- \"How can I customize my UI Canvas buttons, considering that I'm working on a Unity XRI Project?\""
    REAL_QUESTION_EXAMPLES += "\n- \"How can I clip a GameObject with two or more clipping objects?\""
    REAL_QUESTION_EXAMPLES += "\n- \"How to set up XRI in my Unity scene?\""
    REAL_QUESTION_EXAMPLES += "\n- \"How do I intercept when a Controller Button is pressed?\""

    ROLE = f"You are a Human Developer expert in {DOMAIN_FULL_NOME}."

    TARGET = f"Users are Unity developers who have never used {DOMAIN_SHORT_NOME} and want to learn it with your help."         
    GENERIC_JOB_ANSWER_USER_QUESTIONS = "Your goal is to answer user questions."
    GENERIC_JOB_GENERATE_SYNTHETIC_DATASET = f"Your goal is to generate a synthetic dataset about {DOMAIN_SHORT_NOME}."
class FrameworkPrompts():
    @classmethod
    def GET_ROLE_PROMPT(cls)->str:
        base = DomainInfo.ROLE + \
               " " + DomainInfo.GENERIC_JOB_ANSWER_USER_QUESTIONS + \
               " " + DomainInfo.TARGET
        base += "\nIf users ask a non-solvable problem, admit it without inventing anything."
        base += "\n\nFollow these steps to answer the user queries."
        base += f"\nCase 1) I provide you {DomainInfo.DOMAIN_SHORT_NOME} Knowledge below. It's extremely important that you relay on those context. You must NOT use your base knowledge."
        base += f"\nCase 2) I don't provide you {DomainInfo.DOMAIN_SHORT_NOME} Knowledge below. In this case, you can relay on your base knowledge but you have to explicitly say it to the user."
        return base
    
class LoaderPrompts():
    @classmethod
    def GET_TYPE1_PROMPT(cls, document_content:str)->tuple[str, callable]:
        system  = DomainInfo.ROLE + \
            " " + DomainInfo.GENERIC_JOB_GENERATE_SYNTHETIC_DATASET + \
            " " + DomainInfo.TARGET
        
        system += f"\n\nI'm interested in questions that are related to {DomainInfo.DOMAIN_SHORT_NOME} and can be answered with the documentation,"
        system += " but the user doesn't know the existence of the documentation and doesn't have access to it."
        system += " This means that:" + \
                "\n 1) You should not generate the question if the answer is not covered in the documentation." + \
                "\n 2) You should not answer the question with a reference to the documentation."
        
        prompt = system
        prompt += "\n\nHere are some examples of real user questions, you will be judged on how well you match this distribution:"
        prompt += DomainInfo.REAL_QUESTION_EXAMPLES

        prompt += "\n\n<DOCUMENT>\n"
        prompt += document_content.strip()
        prompt +="\n</DOCUMENT>"

        prompt += "\n\nYou will now generate a user question and the corresponding answer based on the above document."
        prompt += "\n1) Explain the user context and what problems they might be trying to solve."
        prompt += "\n2) Generate the user question."
        prompt += "\n3) Provide the accurate answer in markdown format to the user question using the documentation."
        prompt += "\n\nYou'll be evaluated on:"
        prompt += "\n- How realistic is that this question could be asked by a human developer?"
        prompt += "\n- Can the question be answered with the information in the document?"
        prompt += "\n- How accurate is the answer? Remember that the user is a human developer, so the answer should be written in a way that a human developer can understand."""
        
        prompt += "\n\nYour output must be in the format below:"""
        prompt += "\nUser Context: <Generate the context>"""
        prompt += "\nUser Question: <Generate the question>"""
        prompt += "\nAnswer: <Generate the answer>"""

        def str_2_qa(original_content:str, question:str, gpt_output:str):
            context_pattern = re.compile(r"User Context:(.*?)(?=User Question:|$)", re.DOTALL)
            question_pattern = re.compile(r"User Question:(.*?)(?=Answer:|$)", re.DOTALL)
            answer_pattern = re.compile(r"Answer:(.*)", re.DOTALL)

            # Extract User Context, User Question, and Answer
            user_context_match = context_pattern.search(gpt_output)
            user_question_match = question_pattern.search(gpt_output)
            answer_match = answer_pattern.search(gpt_output)

            # Check if matches are found before extracting
            user_context = user_context_match.group(1).strip() if user_context_match else ""

            user_question = user_question_match.group(1).strip() if user_question_match else ""
            user_question = user_question[1:-1] if user_question.startswith('"') and user_question.endswith('"') else user_question

            answer = answer_match.group(1).strip() if answer_match else ""

            return user_question, answer
        return prompt, str_2_qa

    @classmethod
    def get_type2_prompt(cls):
        raise NotImplementedError("We don't need this (yet?)")
    
    @classmethod    
    def GET_TYPE3_PROMPT(cls, question:str, n_replies:int, replies_stringified:str):
        prompt  = DomainInfo.ROLE + \
            " " + DomainInfo.GENERIC_JOB_GENERATE_SYNTHETIC_DATASET + \
            " " + DomainInfo.TARGET
        
        prompt += "\n\nI'm going to pass you a question made in a QA forum."
        prompt += f"\nThere are {n_replies} replies."

        prompt += "\n\nYour job is to generate a unified answer based on the question and the comments."
        prompt += "\nJudge yourself if the comments are useful or not. Try to keep the answer as concise as possible."

        prompt += "\n\nYou have to write only the answer, not the question or the comments. Do not write any preliminary text, just your answer."
        prompt += "\nFormat your answer as if you were writing a Markdown document."

        prompt += "\n\nDO NOT:"
        prompt += "\n- Insert any names or references to the comments."
        prompt += "\n- Write any offensive or inappropriate content, even if they were present in the original comments."
        prompt += "\n- Write any irrelevant content."
        
        prompt += "\n\n<QUESTION>\n"
        prompt += question.strip()
        prompt += "\n</QUESTION>"

        prompt += "\n\n<HUMAN ANSWERS>\n"
        prompt += replies_stringified.strip()
        prompt += "\n</HUMAN ANSWERS>"

        return prompt

class ChatPromptTemplates():
    @classmethod
    def CHATBOT_RETRIEVER__PROMPT_TEMPLATE(cls, has_memory:bool, extra_prompt_context:str)->PromptTemplate:
        prompt = extra_prompt_context.strip()+"\n\n" if extra_prompt_context else ""

        prompt += f"The following documents are the most relevant answers to the user's question, sourced from an external Q&A Knowledge Database."
        
        prompt += "\n\n<RELEVANT ANSWERS FROM KNOWLEDGE DATABASE>\n"
        prompt += "{context}"
        prompt += "\n</RELEVANT ANSWERS FROM KNOWLEDGE DATABASE>"
        
        prompt += "\n\n<QUESTION>\n" 
        prompt += "{question}"
        prompt += "\n</QUESTION>"
    
        input_variables = ["context", "question"]
        if has_memory:
            prompt += "\n\nThe history of the conversation so far is the following:"
            prompt += "\n<HISTORY>\n"
            prompt += "{chat_history}"
            prompt += "\n</HISTORY>"
            input_variables.append("chat_history")

        prompt += "\n\nHelpful Answer:"
        return TemplateHelper.generate_prompt_template(input_variables=input_variables, template=prompt)
    

    @classmethod
    def CHATBOT_SIMPLE__PROMPT_TEMPLATE(cls, has_memory:bool, extra_prompt_context:str)->PromptTemplate:
        prompt = extra_prompt_context.strip()+"\n\n" if extra_prompt_context else ""
        prompt += "<QUESTION>\n" 
        prompt += "{question}"
        prompt += "\n</QUESTION>"
    
        input_variables = ["question"]
        if has_memory:
            prompt += "\n\nThe history of the conversation so far is the following:"
            prompt += "\n<HISTORY>\n"
            prompt += "{chat_history}"
            prompt += "\n</HISTORY>"
            input_variables.append("chat_history")

        prompt += "\n\nHelpful Answer:"

        return TemplateHelper.generate_prompt_template(input_variables=input_variables, template=prompt)
        

class JudgePrompts():
    @classmethod
    def GET_ANONYM_PAIRWISE_SCORE_PROMPT(cls, real_question:str, real_answer:str, fst_gen_answer:str, snd_gen_answer:str)->tuple[str, callable]:
        prompt = f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below, considering the real human answer as the reference point. Your goal is to choose the assistant that follows the user's instructions and answers the user's question better, while also comparing their responses against the human answer. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.

    Begin your evaluation by comparing the two AI responses to each other, and then to the human answer to determine which assistant provided the most helpful and accurate response overall. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses or specific assistant names to influence your evaluation. Be as objective as possible.

    It's mandatory to end with a final verdict that clearly states which assistant's response you believe is better based on the factors above. The last line of your response should be the output "1" if Assistant A's answer is better, "2" if Assistant B's answer is better, or "0" if it is a tie.
    
    <=== Score ===>
    After providing your explanation, output your final verdict by strictly following this format:
    Output "1" if Assistant A's answer is better based on the factors above.
    Output "2" if Assistant B's answer is better based on the factors above.
    Output "0" if it is a tie.
    <=== End of Score ===>

    <=== Begin of Example #1 ===>
    I think that Assistant A's answer is better because blablabla.

    Output "1".
    <=== End of Example #1 ===>
    
    <=== Begin of Example #2 ===>
    I think that Assistant B's answer is better because blablabla.

    Output "2".
    <=== End of Example #2 ===>

    <=== Begin of Example #3 ===>
    I think that there is not a better answer that the other because blablabla.

    Output "0".
    <=== End of Example #3 ===>

    <=== Human ===>
    [The Start of User Question] {real_question} [The End of User Question] 

    [The Start of Human's Answer] {real_answer} [The End of Human's Answer]

    [The Start of Assistant A's Answer] {fst_gen_answer} [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer] {snd_gen_answer} [The End of Assistant B's Answer]
    <=== End of Human ===>

    Which assistant answer is preferred?
    """
        
        def str2value(input_text):
            # Define the regex pattern to match the output "(\d)"
            pattern = r'Output\s+"(\d)"'
            
            # Search for the pattern in the input text
            match = re.search(pattern, input_text)
            
            # If a match is found, return the first captured group (the digit)
            if match:
                r = match.group(1)
                if r in ["0", "1", "2"]:
                    return r
                raise ValueError(f"Invalid output: {r}")
            else:
                raise ValueError(f"Invalid No match found in the AI answer")
   
        return prompt, str2value

    @classmethod
    def GET_RATE_1__5_PROMPT(cls, real_question:str, real_answer:str, gen_answer:str)->tuple[str, callable]:
        prompt = f"""Please act as an impartial judge and evaluate the quality of a response provided by an AI assistant to the user question displayed below, considering the real human answer as the reference point. Your goal is to give a score from 1 to 5 to the assistant answer considering how well it answers the user's question, while also comparing the response against the human answer. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.

    Begin your evaluation by comparing the AI response to the human answer to determine how much the assistant response was helpful and accurate overall. Avoid any style biases. Do not allow the length of the response to influence your evaluation. Be as objective as possible.

    It's mandatory to end with a final verdict that clearly states which assistant's response you believe is better based on the factors above. The last line of your response should be the output followed by a number between 1 to 5, with 1 the worst and 5 the best.
    
    After providing your explanation, output your final verdict by strictly following this format:
    - Score "5": No hallucination, all information is accurate and supported by the context.
    - Score "4": Minor hallucination, mostly accurate with slight embellishments.
    - Score "3": Moderate hallucination, mix of accurate and false information.
    - Score "2": Significant hallucination, mostly false or unsupported information.
    - Score "1": Complete hallucination, entirely false or unrelated to the context.

    
    <=== Example #1 ===>
    I think that Assistant's answer is perfect because blablabla.

    Score "5".
    <=== End of Example #1 ===>
    
    <=== Example #2 ===>
    I think that Assistant's answer is completely wrong because blablabla.

    Score "1".
    <=== End of Example #2 ===>

    <=== Human ===>
    [The Start of User Question] {real_question} [The End of User Question] 

    [The Start of Human's Answer] {real_answer} [The End of Human's Answer]

    [The Start of Assistant's Answer] {gen_answer} [The End of Assistant's Answer]
    <=== End of Human ===>

    How much the assistant score?
    """
        ############################################################
        
        def str2int_score(input_text):
            # Define the regex pattern to match the output "(\d)"
            pattern1 = r'Score\s+"?(\d)"?'
            
            # Search for the pattern in the input text
            match1 = re.search(pattern1, input_text)
            
            # If a match is found, return the first captured group (the digit)
            if match1:
                r = match1.group(1)
                if r in ["1", "2", "3", "4", "5"]:
                    # Return the string as integer
                    return int(r)
                
                raise ValueError(f"Invalid output: {r}")
            else:
                pattern2 = r'Score\s?:?\s?(?:===>)?\s+?"?(\d)"?'
                match2 = re.search(pattern2, input_text)
                if match2:
                    r = match2.group(1)
                    if r in ["1", "2", "3", "4", "5"]:
                        # Return the string as integer
                        print("WARNING: Found the score in the second pattern")
                        return int(r)
                    raise ValueError(f"Invalid output: {r}")
                raise ValueError(f"Invalid No match found in the AI answer. Answer: {input_text}")
   
        return prompt, str2int_score