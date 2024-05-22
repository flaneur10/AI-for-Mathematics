from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
from jsonformer import Jsonformer
import numpy as np
import torch
import json
import re

model_id = "tuned_model/gemma-it-qlora-mathinstruct/"
bert_id = "sentence-transformers/all-MiniLM-L6-v2"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "google/gemma-7b-it"
Questions_dir = "knowledge_dataset/questions-2.json"
KG_dir = "knowledge_dataset/s_math_entity_3new.jsonl"
VD_dir = "knowledge_dataset/sbert_vector_math_entity_3new.jsonl"
output_dir = "output/result.txt"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    device_map="auto"
)

bert_tokenizer = BertTokenizer.from_pretrained(bert_id)
bert_model = BertModel.from_pretrained(bert_id, device_map="cpu")

theory_adapter = "/theory_adapter"
sym_adapter = "/pysym_adapter"
application_adapter = "/application_adapter"

model.load_adapter(model_id + sym_adapter, adapter_name="pysym_adapter")
model.set_adapter("pysym_adapter")

model.load_adapter(model_id + sym_adapter, adapter_name="theory_adapter")
model.set_adapter("theory_adapter")

model.load_adapter(model_id + application_adapter, adapter_name="application_adapter")
model.set_adapter("application_adapter")

model.disable_adapters()

def write_result(output_dir, output):
    with open(output_dir, "a") as file:
        file.writelines(output)

def getquestion(Questions):
#    question = "You want to put 5 different books A, B, C, D, and E on a bookshelf. When A, B, and C are placed on the left side in such order, find the number of ways to place books on the bookshelf."
#    question = "Passage: According to CBS, in 2001 the ethnic makeup of the city was 99.8% Jewish and other non-Arab, without significant Arab population. See Population groups in Israel. According to CBS, in 2001 there were 23,700 males and 24,900 females. The population of the city was spread out with 31.4% 19 years of age or younger, 15.7% between 20 and 29, 18.5% between 30 and 44, 18.3% from 45 to 59, 4.1% from 60 to 64, and 11.9% 65 years of age or older. The population growth rate in 2001 was 0.8%. Question: How many percent were not under the age of 19?"
#    question = r"What is the maximum and minimum of function f(x) = \frac{sin(x)}{x} - e^{-x^2}."
    question = "Suppose there is a box containing 3 red balls and 5 blue balls, making a total of 8 balls. Two balls are randomly drawn from this box without replacement.Calculate the probability of drawing at least one red ball in any of the two draws."

    return question


def chat_format(role, content):
    chat_format_sentence = {"role": role, "content": content}
    return chat_format_sentence


def bert_embedding(sentence, bert_model, bert_tokenizer):
    inputs1 = bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    # Get sentence embeddings
    with torch.no_grad():
        outputs1 = bert_model(**inputs1)

    # Get the sentence embedding (pooling)
    s1_embedding = outputs1.pooler_output
    s1_embedding = s1_embedding.cpu().numpy()
    return s1_embedding

def embedding_sentence_2(title, type, contents, filed, bert_model, bert_tokenizer):
    weight_list = [0.3, 0.1, 0.6]
    title_sentence = "The title is " + type + ": " + title + "."
    contents_sentence = "The content is " + contents + "."
    filed_sentence = "The field is " + filed + "."
    title_embedding = bert_embedding(title_sentence, bert_model, bert_tokenizer)
    contents_embedding = bert_embedding(contents_sentence, bert_model, bert_tokenizer)
    filed_embedding = bert_embedding(filed_sentence, bert_model, bert_tokenizer)

    weight_embedding = weight_list[0] * title_embedding + weight_list[1] * contents_embedding + weight_list[2] * filed_embedding
    return weight_embedding

def embedding_sentence(title, type, contents, filed, bert_model, bert_tokenizer):
    title_sentence = "The title is " + type + ": " + title + "."
    contents_sentence = "The content is " + contents + "."
    filed_sentence = "The field is " + filed + "."

    weight_embedding = bert_embedding(title_sentence + contents_sentence + filed_sentence, bert_model, bert_tokenizer)
    return weight_embedding


def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


# Cosine distance
def find_nearest_word(x, VD, s_VD):
    """Find the word in VD with the highest cosine similarity to x."""
    similarities = {}
    for entity in VD:
        # choose the distance metrics
        similarities[entity["entity contents"]["title"]] = {
                        "similarity": cosine_similarity(x, entity["vector"]),
                        "type": entity["entity contents"]["type"]}
    s_similarities = {}
    for entity in s_VD:
        # choose the distance metrics
        s_similarities[entity["entity contents"]["title"]] = {
                        "similarity": cosine_similarity(x, entity["vector"]),
                        "type": entity["entity contents"]["type"]}

    return similarities, s_similarities


def find_min_similarities_by_type(s_similarities, similarities, knowledge_type):
    min_similarities_by_type = {}

    s_similarities_list = []
# s_similarity search:
    for title, details in s_similarities.items():
        similarity = details["similarity"]
        type_ = details["type"]
        s_similarities_list.append(similarity)
    s_similarities_list.sort(reverse=True)
    if len(s_similarities_list) >= 10:
        for title, details in s_similarities.items():
            similarity = details["similarity"]
            type_ = details["type"]
            if similarity >= s_similarities_list[9]:
                min_similarities_by_type[title] = similarity
    else:
        rest = 10 - len(s_similarities_list)
        for title, details in s_similarities.items():
            similarity = details["similarity"]
            type_ = details["type"]
            min_similarities_by_type[title] = similarity

        similarities_list = {"problem": [],
                             "others": []}

    # 遍历 similarities 字典
        for title, details in similarities.items():
            similarity = details["similarity"]
            type_ = details["type"]
            if type_ == "problem":
                similarities_list["problem"].append(similarity)
            else:
                similarities_list["others"].append(similarity)
            # 如果当前相似度值小于当前类型已记录的最小相似度，则更新记录
        if knowledge_type == "problem":
            similarities_list["problem"].sort(reverse=True)
            for title, details in similarities.items():
                similarity = details["similarity"]
                type_ = details["type"]
                if similarity >= similarities_list["problem"][rest]:
                    if type_ == "problem":
                        min_similarities_by_type[title] = similarity
        else:
            similarities_list["others"].sort(reverse=True)
            for title, details in similarities.items():
                similarity = details["similarity"]
                type_ = details["type"]
                if similarity >= similarities_list["others"][rest]:
                    if type_ != "problem":
                        min_similarities_by_type[title] = similarity

    # 输出每个类型中具有最小相似度值的标题
    # for title, similarity in min_similarities_by_type.items():
    #     print(f"For title: '{title}', the similarity is '{similarity}'")
    return min_similarities_by_type


def small_VD(key_word, VD):
    s_VD = []
    for entity in VD:
        # choose the distance metrics
        pattern = re.compile(re.escape(key_word), re.IGNORECASE)
        if entity["entity contents"]["label"] != "prblem":
            if any(
                    pattern.search(entity["entity contents"]["title"]) or
                    pattern.search(contents) for contents in entity["entity contents"]["contents"]
            ):
                s_VD.append(entity)
    # if the related knowledge is too small
    return s_VD


def question_VD(KG_answer, VD):
    s_VD = []
    for entity in VD:
        # choose the distance metrics
        key_word_1 = KG_answer["knowledge_1 key word"]
        key_word_2 = KG_answer["knowledge_2 key word"]
        key_word_3 = KG_answer["knowledge_3 key word"]

        pattern_1 = re.compile(re.escape(key_word_1), re.IGNORECASE)
        pattern_2 = re.compile(re.escape(key_word_2), re.IGNORECASE)
        pattern_3 = re.compile(re.escape(key_word_3), re.IGNORECASE)
        if entity["entity contents"]["label"] == "prblem":
            if any(
                    pattern_1.search(entity["entity contents"]["title"]) or
                    pattern_2.search(entity["entity contents"]["title"]) or
                    pattern_3.search(entity["entity contents"]["title"]) or
                    pattern_1.search(contents) or
                    pattern_2.search(contents) or
                    pattern_3.search(contents) for contents in entity["entity contents"]["contents"]
            ):
                s_VD.append(entity)

    return s_VD


def extract_knowledge(knowledge_1_title, knowledge_1_type, min_similarities_by_type, KG, pre_answer, file):
    Auxiliary_knowledge_1 = ""

    # use the LLM to tell which knowledge to choose

    json_schema = {
        "type": "object",
        "properties": {}
    }

    titles_list = []
    for titles in min_similarities_by_type:
        json_schema["properties"][titles] = {"type": "boolean"}

    prompt = "Question: " + file + "\nThis is a problem of " + pre_answer["problem type"]  + "\n" + \
             "The problem feild is " + pre_answer["knowledge field"] + "\n" + \
             "Before solve this question, your searched in Math database for " + knowledge_1_title + \
             ". The following is the result back.\n" + \
             "You need to tell whether the knowledge points is related to this question.\n"
    # prompt = "One of the knowledge title this question need is : " + knowledge_1_title + "\n" + \
    #          "You need to tell which knowledge points are related to this question.\n"
    i = 1
    for titles in min_similarities_by_type:
        prompt = prompt + "The title of  knowledge_" + str(i) + " is " + titles + ".\n"
    prompt = prompt + "True means it is related, False means it is not."

    jsonformer_knowledge = Jsonformer(model, tokenizer, json_schema, prompt)
    knowledge_need_list = jsonformer_knowledge()
    # print(knowledge_need_list)
    knowledge_list = []
    for knowledge in knowledge_need_list:
        if knowledge_need_list[knowledge]:
            knowledge_list.append(1)
        else:
            knowledge_list.append(0)
    i = 0
    knowledge_title_list = []
    for titles in min_similarities_by_type:
        if knowledge_list[i] == 1:
            knowledge_title_list.append(titles)
        i += 1

    # extract the corresponding knowledge
    for entity in KG:
        # extract the theorem and definition knowledge
        if knowledge_1_type == "Theorem" or knowledge_1_type == "Definition":
            if entity["title"] in knowledge_title_list:
                knowledge_1_contents = " ".join(entity["contents"])
                knowledge_1_ref = " ".join(f"{ref}. " for ref in entity["refs"])
                if entity["proofs"] != []:
                    knowledge_1_proof = " ".join(
                        [f"({action}): {desc}\n" for item in entity["proofs"][0]["bodylist"] for desc, action in
                        [(item["description"], item["action"])]])
                else: knowledge_1_proof = "Not find."
                Auxiliary_knowledge_1 = "Theorem: " + str(entity["title"]) + \
                                        " Content: " + knowledge_1_contents
                if pre_answer["problem type"] == "Proof":
                    Auxiliary_knowledge_1 = Auxiliary_knowledge_1 + " Proof: " + knowledge_1_proof + " Reference: " + knowledge_1_ref
                Auxiliary_knowledge_1 = Auxiliary_knowledge_1 + "\n"

        # extract the problem knowledge
        if knowledge_1_type == "problem":
            if entity["title"] in knowledge_title_list:
                problem_contents = " ".join(entity["contents"])
                problem_solutions = " ".join(
                    [content for solution in entity["solutions"] for content in solution["contents"]])

                Auxiliary_knowledge_1 = "Example problem: " + problem_contents + "\nSolution: " + problem_solutions + "\n"
    return Auxiliary_knowledge_1


def KG_search(pre_answer, KG, VD, file):
    json_schema = {
        "type": "object",
        "properties": {
            "knowledge_1 key word": {"type": "string"},
            "knowledge_1 title": {"type": "string"},
            "knowledge_1 type": {"type": "string"},
            "knowledge_2 key word": {"type": "string"},
            "knowledge_2 title": {"type": "string"},
            "knowledge_2 type": {"type": "string"},
            "knowledge_3 key word": {"type": "string"},
            "knowledge_3 title": {"type": "string"},
            "knowledge_3 type": {"type": "string"},
            "problem title": {"type": "string"},
        }
    }

    prompt = "Problem: " + file + "\n" + \
             "The Knowledge points: " + pre_answer["Knowledge points"] + "\n" + \
             "You are going to search in Math knowledge database, Please give me the three key words for searching.\n" + \
             "What is the knowledge_1 key word?\n" + \
             "What is the knowledge_1 title that is needed to solve this question?\n" + \
             "Is the knowledge_1 a theorem or a definition?\n" + \
             "What is the knowledge_2 key word?\n" + \
             "What is the knowledge_2 title that is needed to solve this question?\n" + \
             "Is the knowledge_2 a theorem or a definition?\n" + \
             "What is the knowledge_3 key word?\n" + \
             "What is the knowledge_3 title that is needed to solve this question?\n" + \
             "Is the knowledge_3 a theorem or a definition?\n" + \
             "Please give this question a title.\n" + \
             "You need to answer these questions based on the following schema."

    jsonformer_KG = Jsonformer(model, tokenizer, json_schema, prompt)
    KG_answer = jsonformer_KG()
    print("KG_answer: ", KG_answer)
    write_result(output_dir, KG_answer)

    # offer the premilitery knowledge contents
    json_schema = {
        "type": "object",
        "properties": {
            "knowledge_1 contents": {"type": "string"},
            "knowledge_2 contents": {"type": "string"},
            "knowledge_3 contents": {"type": "string"},
            "problem title": {"type": "string"},
        }
    }

    prompt = "What is the contents of " + KG_answer["knowledge_1 title"] + "?\n" + \
             "What is the contents of " + KG_answer["knowledge_2 title"] + "?\n" + \
             "What is the contents of " + KG_answer["knowledge_3 title"] + "?\n" + \
             "You need to answer these questions based on the following schema."

    jsonformer_contents = Jsonformer(model, tokenizer, json_schema, prompt)
    KG_contents = jsonformer_contents()
    print("KG_contents: ", KG_contents)
    write_result(output_dir, KG_contents)

    # Exact search
    s1_VD = small_VD(KG_answer["knowledge_1 key word"], VD)
    s2_VD = small_VD(KG_answer["knowledge_2 key word"], VD)
    s3_VD = small_VD(KG_answer["knowledge_3 key word"], VD)
    s4_VD = question_VD(KG_answer, VD)


    # Fuzzy search

    # Get the sentence embedding
    s1_embedding = embedding_sentence(KG_answer["knowledge_1 title"], KG_answer["knowledge_1 type"], KG_contents["knowledge_1 contents"], pre_answer["knowledge field"], bert_model, bert_tokenizer)
    s2_embedding = embedding_sentence(KG_answer["knowledge_2 title"], KG_answer["knowledge_2 type"], KG_contents["knowledge_2 contents"], pre_answer["knowledge field"], bert_model, bert_tokenizer)
    s3_embedding = embedding_sentence(KG_answer["knowledge_3 title"], KG_answer["knowledge_3 type"], KG_contents["knowledge_3 contents"], pre_answer["knowledge field"], bert_model, bert_tokenizer)
    s4_embedding = embedding_sentence(KG_answer["problem title"], "problem", file, pre_answer["knowledge field"], bert_model, bert_tokenizer)

    # search for the relative word in VD
    similarities_1, s_similarities_1 = find_nearest_word(s1_embedding, VD, s1_VD)
    similarities_2, s_similarities_2 = find_nearest_word(s2_embedding, VD, s2_VD)
    similarities_3, s_similarities_3 = find_nearest_word(s3_embedding, VD, s3_VD)
    similarities_4, s_similarities_4 = find_nearest_word(s4_embedding, VD, s4_VD)

    # retrieve the auxiliary
    min_similarities_by_type_1 = find_min_similarities_by_type(s_similarities_1, similarities_1, KG_answer["knowledge_1 type"])
    min_similarities_by_type_2 = find_min_similarities_by_type(s_similarities_2, similarities_2, KG_answer["knowledge_2 type"])
    min_similarities_by_type_3 = find_min_similarities_by_type(s_similarities_3, similarities_3, KG_answer["knowledge_3 type"])
    min_similarities_by_type_4 = find_min_similarities_by_type(s_similarities_4, similarities_4, "problem")

    # extract knowledge_1
    Auxiliary_knowledge_1 = extract_knowledge(KG_answer["knowledge_1 title"], KG_answer["knowledge_1 type"], min_similarities_by_type_1, KG, pre_answer, file)

    # extract knowledge_2
    Auxiliary_knowledge_2 = extract_knowledge(KG_answer["knowledge_2 title"], KG_answer["knowledge_2 type"], min_similarities_by_type_2, KG, pre_answer, file)

    # extract knowledge_3
    Auxiliary_knowledge_3 = extract_knowledge(KG_answer["knowledge_3 title"], KG_answer["knowledge_3 type"], min_similarities_by_type_3, KG, pre_answer, file)

    # extract example problem
    Auxiliary_knowledge_4 = extract_knowledge(KG_answer["problem title"], "problem", min_similarities_by_type_4, KG, pre_answer, file)

    Auxiliary_knowledge = Auxiliary_knowledge_1 + Auxiliary_knowledge_2 + Auxiliary_knowledge_3 + Auxiliary_knowledge_4

    return Auxiliary_knowledge


def Load_Dataset(save_dir):
    with open(save_dir, 'r') as database_file:
        Dataset = []
        vector_list = database_file.readlines()
        for data in vector_list:
            data_dic = json.loads(data)
            Dataset.append(data_dic)
    return Dataset

def Load_questions(save_dir):
    with open(save_dir, 'r', encoding='utf-8') as f:
        Questions = json.load(f)
    return Questions

# load professional database
KG = Load_Dataset(KG_dir)
VD = Load_Dataset(VD_dir)
Questions = Load_questions(Questions_dir)

def question_solve(initial_information, file, havesolve):
    ### Ask the classification of question
    ### extract the knowledge point

    # jsonformer way

    json_schema = {
        "type": "object",
        "properties": {
            "problem type": {"type": "string"},
            "Knowledge points": {
                "type": "string"
            },
            "knowledge field": {"type": "string"},
            "application_adapter": {"type": "boolean"},
            "pysym_adapter": {"type": "boolean"},
            "theory_adapter": {"type": "boolean"}
        }
    }

    prompt = "Problem: " + file + \
             "You need to follow the guidance to answer the question.\n" + \
             "Application problems involve using theory in real-world scenarios, calculation problems require solving mathematical computations, and proof problems involve demonstrating the validity of a statement through logical arguments.\n" + \
             "Is this a problem of application, proof or calculation? \n" + \
             "Here is a math knowledge basement, you can search knowledge points in it." + \
             "What knowledge points are needed to solve this problem? Please list them.\n" + \
             "What knowledge field are knowledge points? \n" + \
             "Application_adapter can help solving application problem. Do you need application_adapter for this problem? yes or no.\n" + \
             "Pysym_adapter can help in calculation. Do you need pysym_adapter for this problem? yes or no.\n" + \
             "Theory_adapter can help in math theory and prove. Do you need theory_adapter for this problem? yes or no.\n" + \
             "You need to answer these questions based on the following schema."
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    pre_answer = jsonformer()
    print(pre_answer)
    pre_answer_text = ""
    for key, value in pre_answer.items():
        pre_answer_text = pre_answer_text + " " + key + ": " + str(value) + " "
    write_result(output_dir, pre_answer_text + "\n")

    # search for auxiliary knowledge from KG

    Auxiliary_knowledge = KG_search(pre_answer, KG, VD, file)
    # Auxiliary_knowledge = "Limit: In calculus, a limit is a fundamental concept that describes the behavior of a function or sequence as its input (or index) approaches a particular value or infinity. " \
    #                       "Exponential Function: An exponential function is a mathematical function of the form f(x)=a^x, where a is a positive constant called the base and x is the independent variable." \
    #                       "Sinusoidal Function: A sinusoidal function, also known as a sinusoid, is a periodic function that exhibits a smooth, repetitive oscillation pattern. The most common sinusoidal functions are the sine function sin(x) and cosine function cos(x)."


    ### agent set the adapter
    # set application_adapter
    adapter_fit = ""
    if pre_answer["application_adapter"]:
        model.set_adapter("application_adapter")
        adapter_fit = adapter_fit + "To solve application problem, you need to analyse the question first, then choose proper math algorithm. For complex "\
                "question, you can devide this problem and solve it step by step.\n"

    # set pysym_adapter
    if pre_answer["pysym_adapter"]:
        model.set_adapter("pysym_adapter")
        adapter_fit = adapter_fit + "Python sympy package helps to solve calculation problem. You can write a proper python "\
                "program to solve calculation correctly.\n"

    # set theory_adapter
    if pre_answer["theory_adapter"]:
        model.set_adapter("theory_adapter")
        adapter_fit = adapter_fit + "Proof problem need rigorous derivation and list the math theory supporting derivation."\
        "For complex problem, you can devide the problem and proof step by step.\n"


    ### solve the question

    # Use knowledge to help solve question
    chat_3 = [
        chat_format("user",
                    "Please solve following question.\n" + file + "\n" +\
                    "Here is some information from database:\n" + Auxiliary_knowledge + "\n" + adapter_fit + "\n"
                    + "Question: " + file
                    ),
    ]

    # 对数学问题进行解答
    prompt_3 = tokenizer.apply_chat_template(chat_3, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    outputs_3 = model.generate(prompt_3.to(model.device), max_new_tokens=500)

    #输出解答
    output_text_3 = tokenizer.decode(outputs_3[0])
    write_result(output_dir, output_text_3 + "\n")


    ### check the answer

    json_schema_2 = {
        "type": "object",
        "properties": {
            "Self check": {
                "type": "boolean"
                },
            "Check result": {
                "type": "string"
            },
        }
    }

    prompt_4 = output_text_3 + "\n" + "Please check this answer whether it is correct.\n" + \
               "Here are some guidelines to help you determine if the solution is correct.\n" + \
               "For a proof problem, you need to examine whether the theorems are applied correctly and whether the logical reasoning is correct.\n" +  \
               "For a computational problem, you should check if the calculations are accurate.\n" + \
               "In the case of applied problems, it is necessary to verify if the mathematical model has been constructed appropriately.\n" + \
               "True means correct, False means incorrect.\n" + \
               "If it is incorrect, please give check result about the mistake. Else, please give some results.\n" + \
               "You need to answer these questions based on the following schema."

    jsonformer_2 = Jsonformer(model, tokenizer, json_schema_2, prompt_4)
    final_answer = jsonformer_2()
    final_answer_text = ""
    for key, value in final_answer.items():
        final_answer_text = final_answer_text + " " + key + ": " + str(value) + " "
    write_result(output_dir, final_answer_text + "\n")

    # pass the check
    if final_answer["Self check"]:
        print(output_text_3)
        write_result(output_dir, output_text_3 + "\n")
        havesolve = -1
    else:
        prompt_5 = output_text_3 + "\n" + final_answer["Check result"] + "\n" + "Please correct it"
        inputs_5 = tokenizer(prompt_5, return_tensors="pt").to(model.device)
        outputs_5 = model.generate(**inputs_5, max_new_tokens=500)
        output_text_5 = tokenizer.decode(outputs_5[0])

        prompt_6 = output_text_5 + "\n" + "Please check this answer whether it is correct.\n" + \
               "Here are some guidelines to help you determine if the solution is correct.\n" + \
               "For a proof problem, you need to examine whether the theorems are applied correctly and whether the logical reasoning is correct.\n" +  \
               "For a computational problem, you should check if the calculations are accurate.\n" + \
               "In the case of applied problems, it is necessary to verify if the mathematical model has been constructed appropriately.\n" + \
               "True means correct, False means incorrect.\n" + \
               "If it is incorrect, please give check result about the mistake. Else, please give some results.\n" + \
               "You need to answer these questions based on the following schema."

        jsonformer_3 = Jsonformer(model, tokenizer, json_schema_2, prompt_6)
        final_answer = jsonformer_3()
        final_answer_text = ""
        for key, value in final_answer.items():
            final_answer_text = final_answer_text + " " + key + ": " + str(value) + " "
        write_result(output_dir, final_answer_text + "\n")
        if final_answer["Self check"]:
            #print(output_text_5)
            write_result(output_dir, output_text_5 + "\n")
            havesolve = -1
        else:
            prompt_6 = output_text_3 + "\n" + final_answer["Check result"] + "\n" + "Please summary the mistake."
            inputs_6 = tokenizer(prompt_6, return_tensors="pt").to(model.device)
            outputs_6 = model.generate(**inputs_6, max_new_tokens=150)
            initial_information = tokenizer.decode(outputs_6[0])
            print(initial_information)
            write_result(output_dir, initial_information + "\n")
            havesolve += 1


    return havesolve, initial_information


question = getquestion(Questions)

def Math_AI(question):
    solved = 0
    # record the error information
    initial_information = ""

    while solved >= 0 and solved < 3:
        solved, initial_information = question_solve(initial_information, question, solved)
        print(solved)

    if solved > 0:
        answer = "I can't solve this question, I need more information\n"
        write_result(output_dir, answer + "\n")

for fields in Questions:
    for question in Questions[fields]:
        file = fields + ": " + question
        Math_AI(question)