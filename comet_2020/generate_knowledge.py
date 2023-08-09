import json
import torch
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import use_task_specific_params, trim_batch
from tqdm import tqdm
import sys
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ErcDatasetComet(torch.utils.data.Dataset):
    def __init__(self, DATASET='MELD', SPLIT='train'):
        """Initialize emotion recognition in conversation text modality dataset class."""

        self.DATASET = DATASET
        self.SPLIT = SPLIT
        if self.DATASET in ['MELD', 'IEMOCAP']:
            self._load_emotions()
            self._load_utterance_ordered()
            self.data = self._string2tokens()
        else:
            self.data = self._load_data()

    def _load_emotions(self):
        """Load the supervised labels"""
        with open(os.path.join('./data', self.DATASET, 'emotions.json'), 'r') as stream:
            self.emotions = json.load(stream)[self.SPLIT]

    def _load_data(self):
        """Load data for EmoryNLP dataset."""
        with open(os.path.join('./data', self.DATASET, self.SPLIT+'_data.json'), 'r') as f:
            raw_data = json.load(f)
        return raw_data

    def _load_utterance_ordered(self):
        """Load the ids of the utterances in order."""
        with open(os.path.join('./data', self.DATASET, 'utterance-ordered.json'), 'r') as stream:
            utterance_ordered = json.load(stream)[self.SPLIT]

        count = 0
        self.utterance_ordered = {}
        for diaid, uttids in utterance_ordered.items():
            self.utterance_ordered[diaid] = []
            for uttid in uttids:
                try:
                    with open(os.path.join('./data', self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json'), 'r') as stream:
                        foo = json.load(stream)
                    self.utterance_ordered[diaid].append(uttid)
                except Exception as e:
                    count += 1

    def _load_utterance_speaker_emotion(self, uttid) -> dict:
        """Load an speaker-name prepended utterance and emotion label"""
        text_path = os.path.join(
            './data', self.DATASET, 'raw-texts', self.SPLIT, uttid + '.json')

        with open(text_path, 'r') as stream:
            text = json.load(stream)

        utterance = text['Utterance'].strip()
        emotion = text['Emotion']

        if self.DATASET == 'MELD':
            speaker = text['Speaker']
        elif self.DATASET == 'IEMOCAP':
            sessid = text['SessionID']
            # https: // www.ssa.gov/oact/babynames/decades/century.html
            speaker = {'Ses01': {'Female': 'Mary', 'Male': 'James'},
                       'Ses02': {'Female': 'Patricia', 'Male': 'John'},
                       'Ses03': {'Female': 'Jennifer', 'Male': 'Robert'},
                       'Ses04': {'Female': 'Linda', 'Male': 'Michael'},
                       'Ses05': {'Female': 'Elizabeth', 'Male': 'William'}}[sessid][text['Speaker']]

        else:
            raise ValueError(f"{self.DATASET} not supported!!!!!!")

        return {'text': utterance, 'label': emotion, 'speaker': speaker}

    def _create_input(self, diaids):
        inputs = []
        for diaid in tqdm(diaids):
            ues = [self._load_utterance_speaker_emotion(uttid)
                   for uttid in self.utterance_ordered[diaid]]
            inputs.append(ues)
        return inputs

    def _string2tokens(self):
        """Convert string to (BPE) tokens."""
        diaids = sorted(list(self.utterance_ordered.keys()))

        return self._create_input(diaids=diaids)

class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

def get_csk_feature(model, tokenizer, context, relations):
    features = []
    for id, utterance in enumerate(context):
        queries = []
        for r in relations:
            queries.append("{} {} [GEN]".format(utterance, r))
        with torch.no_grad():
            batch = tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(device)
            input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            activations = out['decoder_hidden_states'][-1][:, 0, :].detach().cpu().numpy()
            features.append(activations)
    return features


def RECCON_get_csk_feature(model, tokenizer, context, relations):
    features = {}
    for id, conv in tqdm(context.items()):
        conv_feature = []
        for utterance in conv:
            queries = []
            for r in relations:
                queries.append("{} {} [GEN]".format(utterance, r))
            with torch.no_grad():
                batch = tokenizer(queries, return_tensors="pt", truncation=True, padding="max_length").to(device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=tokenizer.pad_token_id)
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                activations = out['decoder_hidden_states'][-1][:, 0, :].detach().cpu().numpy()
                conv_feature.append(activations)
        assert len(conv) == len(conv_feature)
        features[id] = conv_feature
    return features

if __name__ == "__main__":
    model_path = "./comet-atomic_2020_BART/"
    dataset_name = 'RECCON'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = str(model.device)
    print(device)
    use_task_specific_params(model, "summarization")
    model.zero_grad()
    model.eval()
    
    batch_size = 8
    #relations_social = ["xReact", "xWant", "xIntent", "oReact", "oWant"]
    #relations_event = ["isAfter", "HasSubEvent", "isBefore", "Causes", "xReason"]
    relations = ['xIntent', 'xAttr', 'xNeed', 'xWant', 'xEffect', 'xReact', 'oWant', 'oEffect', 'oReact', "isAfter", "isBefore"]
    
    for split in ['train', 'val', 'test']:
        if dataset_name != 'RECCON':
            dataset = ErcDatasetComet(dataset_name, split)
            extracted_data = []
            print("\tSplit: {}".format(split))
            for j, dialogue in enumerate(dataset.data):
                inputs = [utt['text'] for utt in dialogue]
                extracted_dialogue = []
                features = get_csk_feature(model, tokenizer, inputs, relations)
                for i, utt in enumerate(dialogue):
                    utt['atomic_features'] = features[i]
                    extracted_dialogue.append(utt)
                extracted_data.append(extracted_dialogue)
                if j % 100 == 0:
                    print("{} dialogues processed.".format(j))
            pickle.dump(extracted_data,
                        open(os.path.join('./bart_comet_enhanced_data', dataset_name, split + '.pkl'), 'wb+'))
        else:
            target_context, speaker, cause_label, emotion, ids = pickle.load(
                open('./RECCON_data/dailydialog_' + split + '.pkl', 'rb'),
                encoding='latin1')
            extracted_data = RECCON_get_csk_feature(model, tokenizer, target_context, relations)
            pickle.dump([target_context, speaker, cause_label, emotion, ids, extracted_data],
                        open('./bart_comet_enhanced_data/RECCON/' + split + '.pkl', 'wb'))

