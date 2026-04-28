
PROCESS1 = '1-initial'
PROCESS2 = '2-invalid'
PROCESS3 = '3-missing'
PROCESS4 = '4-holistic'
PROCESS5 = '5-multipath'
PROCESS6 = '6-subtree_cons'
PROCESS7 = '7-rename_nodes'
PROCESS8 = '8-domain_sum'
PROCESS9 = '9-domain_cls'
PROCESS10 = '10-entcons_sum'
PROCESS11 = '11-entcons_cls'
PROCESS12 = '12-standard'
PROCESS13 = '13-standard_sum'
PROCESS14 = '14-standard_cls'


P_INIT_REL_TREE = """
### Task Instructions:

You are an expert in hierarchical clustering and are good at using your own language power to solve problems instead of calling external tools. 
Given a set of relation labels, follow the steps below to construct a hierarchical taxonomy:

1. Cluster the relation labels based on both their descriptions and names.
2. Summarize each cluster with a concise, accurate, and informative relation name.
3. Repeat the clustering and summarization process for three iterations to refine the hierarchy.

### Output Requirements:

1. The final taxonomy must retain **ALL** original relation labels without altering them.
2. Every original relation label must appear in the final output.
3. The output must be formatted in **JSON**. Here is an example you can refer to:

```json
{
  "Entity Relationships": {
    "Personal Relationships": [
      "relative",
      "father",
      "sibling",
      "stepparent",
      "unmarried partner"
    ],
    "Political Relationships": [
      "member of political party",
      "political alignment",
      "political ideology",
      "allegiance"
    ]
  },
  "Organizational Relationships": {
    "Founding/Ownership": [
      "founded by",
      "owned by",
      "parent organization",
      "has subsidiary"
    ],
    "Affiliations": [
      "member of",
      "affiliation",
      "participant in",
      "political alignment"
    ]
  }
}
```

### Query:

Please construct a hierarchical taxonomy for the following relation labels extracted from WikiData:
[RELATION_WITH_DESC]
""".strip()


P_FIX_MISSING_NODES = """
Given the following taxonomy of relation labels, please place the relation label "[RELATION_LABEL]" (description: "[RELATION_DESC]") in the most suitable position of the 3rd within the taxonomy.
If there is **no suitable position**, refine the taxonomy to accommodate the new relation label. 

**Relation Label Taxonomy:**
[RELATION_TAXONOMY]

### Response Requirements:

1. **To remove** a node from the taxonomy (e.g., located at *A -> B -> C*), respond with:
   `del A -> B -> C`
 
2. **To add** a new node to the taxonomy (e.g., placed at *E -> F -> D*), respond with:
   `add E -> F -> D`

3. Please note that the relation label must be placed at the 3rd layer.

4. **Only provide modifications** to the taxonomy. Do **not** include explanations or additional information.
""".strip()


P_ADJUST_INAPPROPRIATE_CATEGORIZATION = """
### Task Instructions:

You are an expert in the taxonomy of relation labels.
Given a taxonomy, you can refine its structure from a holistic perspective by following these steps:

1. Identify any **illogical or redundant branches** and remove their root nodes.
2. **Reassign the child nodes** of the removed branches to more appropriate locations within the taxonomy.

**Example:**
If the semantics of node *A* fully encompass node *B*, but both are listed as siblings in the taxonomy, node *B* should be removed, and its child nodes should be merged under node *A*.

### Output Requirements:

1. Provide **only** the final version of refined taxonomy; do **not** include intermediate steps.
2. The taxonomy must be in **JSON** format.

### Query:

Please refine the following hierarchical taxonomy of relation labels:
[RELATION_TAXONOMY]
""".strip()


P_MULTI_PATH = """
### Task Instructions:

You are an expert in the taxonomy of relation labels.
Given a hierarchical taxonomy, analyze each leaf node and identify whether a leaf node may also belong to another category.
If a leaf node should be linked to an additional category, add it accordingly.

### Output Requirements:

1. Provide **only** your modifications without explanations or additional information;
2. Format each modification as: "add A -> B -> C"

### Example:

Given taxonomy:
{
  "A": {
    "B": [
      "C", 
      "D", 
    ],
    "E": [
      "F" 
    ]
  }
}
If "D" could also belong under "E", output: "add A -> E -> D". 

### Query:

Please refine the following hierarchical taxonomy of relation labels:
[RELATION_TAXONOMY]
""".strip()


P_SUBTREE_CONSISTENCY = """
You are an expert in relation taxonomy. 
Given a taxonomy that **might have improper categorizations**:
```json
[RELATION_TAXONOMY]
```

Now, consider the following sub-taxonomy extracted from the above taxonomy:
```json
[SUBTREE]
```

### Task:
1. Ensure all the child nodes are correctly placed under the parent node according to the sub-taxonomy. 
2. If the sub-taxonomy has too few child nodes compared to other sub-taxonomies, copy additional nodes from other sub-taxonomies to enrich it.
3. If the sub-taxonomy is needless to refine, output `yes`.
4. Otherwise:
   - Suggest specific corrective actions, such as: [ACTIONS]
   - If there are multiple corrections needed, output all of them.

Provide only the answer, without additional explanation. 
""".strip()


P_SUBTREE_CONS_ACTION_1 = """
      1. Move a node to a more appropriate position (which must already exist).
         Format: `mv "A/B/C" "D/E/C"` (move `C` from `"A/B"` to `"D/E"`).
      2. Copy a node to another appropriate position .
         Format: `cp "A/B/C" "D/E/C"` (copy `C` from `"A/B"` to `"D/E"`)."""


P_SUBTREE_CONS_ACTION_2 = """
      1. Move a node to a more appropriate position (which must already exist).
         Format: `mv "A/B/C" "D/E/C"` (move `C` from `"A/B"` to `"D/E"`).
      2. Copy a node to another appropriate position .
         Format: `cp "A/B/C" "D/E/C"` (copy `C` from `"A/B"` to `"D/E"`).
      3. Remove an inappropriate node.
         Format: `rm "Z/Y/X"` (remove `X` from `"Z/Y"`)."""


P_RENAME_NODES = """
You are an expert in relation taxonomy. 
Analyze the semantic appropriateness of node name "[NODE_NAME]" based on its child nodes: [CHILDREN]. 
If the current name is suitable, output "no". 
Otherwise, suggest a more appropriate name. 
Provide the answer only without additional explanations. 
""".strip()


P_DOMAIN_SUMMARIZATION = """
Analyze the provided relation labels and their descriptions, and group the relation labels into 14-16 broad domains. 

### Requirements: 
1. Each domain should have a short, clear name that reflects its core theme.
2. Ensure that domains do not overlap in meaning. 
3. Provide a brief yet precise description (~15 words) for each domain.
4. Return the result in valid JSON format as shown below.

### JSON Output Format:
```json
{
  "name of domain 1": "description of domain 1", 
  "name of domain 2": "description of domain 2", 
  ...
}
```

### Labels and descriptions:
```json
[RELATION_WITH_DESC]
```

### Output: 
""".strip()


P_DOMAIN_CLASSIFY = """
You are tasked with classifying the relation label "[REL_NAME]" based on its description: "[REL_DESC]".

### Available Domains
Below is a list of domains with their descriptions:
```json
[DOMAIN_WITH_DESC]
```

### Task
Determine which domain(s) best align with "[REL_NAME]".

### Output Format
Provide your answer as a NON-EMPTY JSON array containing the relevant domain names:
```json
["Domain1", "Domain2", ...]
```
""".strip()


P_ENTCONS_SUMMARIZATION = """
Analyze the given relation labels and their descriptions within the domain of [DOMAIN_NAME]. 
Categorize these relations into 8-10 clusters based on entity type constraints.

### Requirements: 
1. Each entity type constraint should have a clear, structured name reflecting its theme. Use a hyphen ('-') to connect entity types (e.g., "person-location", "object-location"). 
2. Return the result in valid JSON format as shown below.

### JSON Output Format:
```json
["EntityTypeA-EntityTypeB", "EntityTypeC-EntityTypeD", ...]
```

### Labels and descriptions:
```json
[RELATION_WITH_DESC]
```

### Output: 
""".strip()


P_ENTCONS_CLASSIFY = """
You are tasked with identifying the entity type constraints of the relation label "[REL_NAME]" based on its description: "[REL_DESC]".

### Available Entity Type Constraints
Below is a list of entity type constraints with their descriptions:
```json
[ENT_TYPE_CONSTRAINTS]
```

### Task
Determine which entity type constraint(s) best align with "[REL_NAME]".

### Output Format
Provide your answer as a NON-EMPTY JSON array containing the relevant entity type constraints:
```json
["Entity Constraint 1", "Entity Constraint 2", ...]
```
""".strip()


# rule, criterion, standard
P_CRITERION_GENERATION = """
### Background
You are an expert in cross-document relation extraction, which aims to identifying predefined relations between entities that appear in different documents. 

### Task
Your task is to analyze and provide several distinct clustering criterion which is beneficial to group these predefined relations based on: 
  - Homogeneity: Relations in the same cluster should be highly similar.
  - Heterogeneity: Different clusters should be as distinct as possible.

### Requirements
1. Review the concept of cross-document relation extraction and carefully analyze the provided relation names and descriptions.
2. Provide 10-12 distinct clustering criteria with concise names (1-2 words). 
3. Choose the top 3 criteria that might yield the most effective clustering results.
4. Return the result in a valid JSON format as shown below. 

### JSON Output Format:
```json
{
  "clustering criteria": {
    "name of criterion 1": {
      "explanation": "explanation of criterion 1", 
      "possible cluster names": [ "cluster name 1", "cluster name 2" ]
    }, 
    "name of criterion 2": {
      "explanation": "explanation of criterion 1", 
      "possible cluster names": [ "cluster name 1", "cluster name 2" ]
    }, 
    ...
  }, 
  "top3 criteria": ["name of criterion i", "name of criterion j", "name of criterion k"]
}
```

### Output:
""".strip()


P_CRITERION_SUM_PREFIX_1 = """
### Task
Analyze the given relation types and their descriptions, and categorize these relations into 10-12 clusters based on their [CRITERION_NAME]s, where "[CRITERION_NAME]" is defined as: "[CRITERION_EXPLANATION]". 
""".strip()


P_CRITERION_SUM_PREFIX_2 = """
### Task
Analyze the given relation types and their descriptions within the [PREV_CRITERION_NAME] of [PREV_CRITERION_INSTANCE]. 
Categorize these relations into 10-12 clusters based on their [CRITERION_NAME]s, where "[CRITERION_NAME]" is defined as: "[CRITERION_EXPLANATION]". 
""".strip()


P_CRITERION_SUM_PREFIX_3 = """
### Task
Analyze the given relation types and their descriptions within the [PREV_CRITERION_NAME] of [PREV_CRITERION_INSTANCE]. 
Categorize these relations into several clusters based on their [CRITERION_NAME]s, where "[CRITERION_NAME]" is defined as: "[CRITERION_EXPLANATION]". 
""".strip()


P_CRITERION_SUM_PREFIXS = [ P_CRITERION_SUM_PREFIX_1, P_CRITERION_SUM_PREFIX_2, P_CRITERION_SUM_PREFIX_3 ]


P_CRITERION_SUMMARIZATION = """
### Requirements: 
1. Each [CRITERION_NAME] should have a **CONCISE, CLEAR and STRUCTURED** name (1-2 words) reflecting its theme (e.g., [CRITERION_EXAMPLES]). 
2. Ensure that [CRITERION_NAME]s do not overlap in meaning, with each covering a single [CRITERION_NAME]. 
3. Ensure that [CRITERION_NAME]s cover **ALL** provided relation types. 
4. Provide a brief yet precise description (~15 words) for each [CRITERION_NAME]. 
5. Return the result in a valid JSON format as shown below. 

### JSON Output Format:
```json
{
  "name of [CRITERION_NAME] 1": "description of [CRITERION_NAME] 1", 
  "name of [CRITERION_NAME] 2": "description of [CRITERION_NAME] 2", 
  ...
}
```

### Relation Types and Descriptions:
```json
[RELATION_WITH_DESC]
```

### Output: 
""".strip()


P_CRITERION_CLASSIFY = """
### Task
You are tasked with analyzing the [CRITERION_NAME] of the relation label "[REL_NAME]" based on its description: "[REL_DESC]".
Determine which [CRITERION_NAME](s) best align with "[REL_NAME]".

### Available [CRITERION_NAME]s
Below is a list of [CRITERION_NAME]s with their descriptions:
```json
[CRITERION_INSTANCES]
```

### Output Format
Provide your answer as a NON-EMPTY JSON array containing the [CRITERION_NAME](s):
```json
["[CRITERION_NAME] 1", "[CRITERION_NAME] 2", ...]
```

### Output: 
""".strip()

