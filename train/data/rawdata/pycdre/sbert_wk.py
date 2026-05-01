import random
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead


class SentenceEmbedding:

    def __init__(self, embed_method, masks):
        # Select from embedding methods
        switcher = {
            'ave_last_hidden': self.ave_last_hidden,
            'CLS': self.get_cls,
            'ave_one_layer': self.ave_one_layer,
        }
        self.masks = masks
        self.embed = switcher.get(embed_method, 'Not a valide method index.')

    def ave_last_hidden(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - \
            1  # Not considering the last item

        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1, :, :]
            embedding.append(np.mean(hidden_state_sen[:sent_len, :], axis=0))

        embedding = np.array(embedding)
        return embedding

    def ave_one_layer(self, params, all_layer_embedding):
        """
            Average the output from last layer
        """
        unmask_num = np.sum(self.masks, axis=1) - \
            1  # Not considering the last item

        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][params['layer_start'], :, :]
            embedding.append(np.mean(hidden_state_sen[:sent_len, :], axis=0))

        embedding = np.array(embedding)
        return embedding

    def get_cls(self, params, all_layer_embedding):
        """
            CLS vector as embedding
        """
        unmask_num = np.sum(self.masks, axis=1) - \
            1  # Not considering the last item

        embedding = []
        for i in range(len(unmask_num)):
            sent_len = unmask_num[i]
            hidden_state_sen = all_layer_embedding[i][-1, :, :]
            embedding.append(hidden_state_sen[0])

        embedding = np.array(embedding)
        return embedding


class SentenceBert:

    def __init__(self, pretrained_bert_path, device=None):
        self.device = device
        self.model_path = pretrained_bert_path
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelWithLMHead.from_pretrained(self.model_path, config=self.config)
        for param in self.model.parameters():  # frozen everything
            param.requires_grad = False
        if device:
            self.model = self.model.to(device)
        else:
            self.model = self.model.cuda()
        self.max_seq_length = 128
        self.embed_method = 'ave_last_hidden'

    def pair_sims(self, sentence1, sentence2):
        embedding = self.sent2vec([sentence1, sentence2])
        similarity = (
            embedding[0].dot(embedding[1])
            / np.linalg.norm(embedding[0])
            / np.linalg.norm(embedding[1])
        )
        # print("The similarity between these two sentences are (from 0-1):", similarity)
        return similarity

    def sent2vec(self, sentences):
        sentences_index = [self.tokenizer.encode(s, add_special_tokens=True) for s in sentences]
        features_input_ids = []
        features_mask = []
        for sent_ids in sentences_index:
            # Truncate if too long
            if len(sent_ids) > self.max_seq_length:
                sent_ids = sent_ids[: self.max_seq_length]
            sent_mask = [1] * len(sent_ids)
            # Padding
            padding_length = self.max_seq_length - len(sent_ids)
            sent_ids += [0] * padding_length
            sent_mask += [0] * padding_length
            # Length Check
            assert len(sent_ids) == self.max_seq_length
            assert len(sent_mask) == self.max_seq_length

            features_input_ids.append(sent_ids)
            features_mask.append(sent_mask)

        features_mask = np.array(features_mask)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
        if self.device:
            batch = [batch_input_ids.to(self.device), batch_input_mask.to(self.device)]
        else:
            batch = [batch_input_ids.cuda(), batch_input_mask.cuda()]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        self.model.zero_grad()

        with torch.no_grad():
            features = self.model(**inputs)[1]

        # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
        # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
        all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

        embed_method = SentenceEmbedding(self.embed_method, features_mask)
        embeddings = embed_method.embed(self.embed_method, all_layer_embedding)
        
        return embeddings


if __name__ == "__main__":
    # -----------------------------------------------
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=64, type=int, help="batch size for extracting features."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--pretrained_bert_path",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained language models. (default: 'bert-base-uncased')",
    )
    parser.add_argument(
        "--embed_method",
        type=str,
        default="ave_last_hidden",
        help="Choice of method to obtain embeddings (default: 'ave_last_hidden')",
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=2,
        help="Topological Embedding Context Window Size (default: 2)",
    )
    parser.add_argument(
        "--layer_start",
        type=int,
        default=4,
        help="Starting layer for fusion (default: 4)",
    )
    parser.add_argument(
        "--tasks", type=str, default="all", help="choice of tasks to evaluate on"
    )
    args = parser.parse_args()

    # -----------------------------------------------
    # Set device
    torch.cuda.set_device(-1)
    device = torch.device("cuda", 0)
    args.device = device

    # -----------------------------------------------
    # Set seed
    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    set_seed(args)
    # Set up logger
    # logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

    # -----------------------------------------------
    # Set Model
    params = vars(args)

    config = AutoConfig.from_pretrained(params["pretrained_bert_path"], cache_dir="./cache")
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(params["pretrained_bert_path"], cache_dir="./cache")
    model = AutoModelWithLMHead.from_pretrained(
        params["pretrained_bert_path"], config=config, cache_dir="./cache"
    )
    model.to(params["device"])

    # -----------------------------------------------

    sentence1 = input("\nEnter the first sentence: ")
    sentence2 = input("Enter the second sentence: ")

    sentences = [sentence1, sentence2]

    print("The two sentences we have are:", sentences)

    # -----------------------------------------------
    sentences_index = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]
    features_input_ids = []
    features_mask = []
    for sent_ids in sentences_index:
        # Truncate if too long
        if len(sent_ids) > params["max_seq_length"]:
            sent_ids = sent_ids[: params["max_seq_length"]]
        sent_mask = [1] * len(sent_ids)
        # Padding
        padding_length = params["max_seq_length"] - len(sent_ids)
        sent_ids += [0] * padding_length
        sent_mask += [0] * padding_length
        # Length Check
        assert len(sent_ids) == params["max_seq_length"]
        assert len(sent_mask) == params["max_seq_length"]

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

    features_mask = np.array(features_mask)

    batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
    batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
    batch = [batch_input_ids.to(device), batch_input_mask.to(device)]

    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
    model.zero_grad()

    with torch.no_grad():
        features = model(**inputs)[1]

    # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to list
    # of (num_hidden_states, seq_len, hidden_dim) for each element in the batch.
    all_layer_embedding = torch.stack(features).permute(1, 0, 2, 3).cpu().numpy()

    embed_method = SentenceEmbedding(params["embed_method"], features_mask)
    embedding = embed_method.embed(params, all_layer_embedding)

    # similarity = (
    #     embedding[0].dot(embedding[1])
    #     / np.linalg.norm(embedding[0])
    #     / np.linalg.norm(embedding[1])
    # )
    # print("The similarity between these two sentences are (from 0-1):", similarity)
