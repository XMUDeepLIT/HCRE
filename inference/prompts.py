

PLAIN_PROMPT_ORIGIN = """Text: [CONTEXT]
Given the above documents, please tell me the relationship between "[HEAD_ENTITY]" and "[TAIL_ENTITY]". 

Option: [OPTIONS]

Answer: """


MS_PROMPT_ORIGIN = """Text: [CONTEXT]
Knowing that the relationship between "[HEAD_ENTITY]" and "[TAIL_ENTITY]" falls under "[PREV_NODES]", please select the most suitable sub-category from the following options based on the given document. 

Options: [OPTIONS]

Answer: """


PLAIN_PROMPT = """#### Task Instruction
You are an expert in Cross-Document Relation Extraction. 
Your task is to determine the most precise relation between the given entities based on the provided context. 

#### Query
Context: [CONTEXT]
Based on the above context, identify the most accurate relation between "[HEAD_ENTITY]" and "[TAIL_ENTITY]" from the following options. 

Options: [OPTIONS]

Relation: """


MS_PROMPT = """#### Task Instruction
You are an expert in Cross-Document Relation Extraction. 
Your task is to determine the most precise relation between the given entities based on the provided context. 

#### Query
Context: [CONTEXT]
Knowing that the relationship between "[HEAD_ENTITY]" and "[TAIL_ENTITY]" falls under "[PREV_NODES]", please select the most suitable relation from the following options based on the given context. 

Options: [OPTIONS]

Relation: """


MS_PROMPT_NO_PREV_ORIGIN = PLAIN_PROMPT_ORIGIN
MS_PROMPT_NO_PREV = PLAIN_PROMPT

