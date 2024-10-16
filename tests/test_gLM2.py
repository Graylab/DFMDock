import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('tattabio/gLM2_650M', torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)

# A contig with two proteins and an inter-genic sequence.
# NOTE: Nucleotides should always be lowercase, and prepended with `<+>`.
sequence = "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aatttaaggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED"

# Tokenize the sequence.
encodings = tokenizer([sequence], return_tensors='pt')

# Extract embeddings.
with torch.no_grad():
    embeddings = model(encodings.input_ids.cuda(), output_hidden_states=True).last_hidden_state

