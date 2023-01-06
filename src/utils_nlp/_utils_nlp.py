'''
NLP utilities
'''

from typing import Dict, List
from spacy import displacy
from datasets import Dataset


__all__ = ['MyNlp']

class MyNlp:
    '''My NLP related utility fns'''

    # Mapping from generic NER tag to its numerical index
    ner_tag2idx = {
        tag : idx for idx, tag 
            in enumerate(['PER', 'ORG', 'LOC', 'MISC'])
    }
    ner_idx2tag = {val:key for key,val in ner_tag2idx.items()}

    @staticmethod
    def print_mem():
        '''Print the RAM used by the current Python process'''
        import psutil
        # Process.memory_info is expressed in bytes, so convert to megabytes
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    @staticmethod
    def print_size(dataset:Dataset):
        '''Print the dataset size of the `dataset` in GB.
        [ss] 12/07/2022. Note: Doesn't seem to work for datasets created from `Dataset.save_to_disk()`
        '''
        size_gb = dataset.dataset_size / (1024**3)
        print(f"Dataset size (cache file) : {size_gb:.2f} GB")
        
    @staticmethod        
    def disp_ents(
        text:str, 
        entities:List[Dict],
        html:bool=False,
    ):
        '''Visualizes the `entities` in this Jupyter notebook.

        Currently only handles huggingface's NER output format (i.e. List[Dict])
        and Flair's `Sentence` format..

        `text` is the raw text string where the `entities` are extracted from.
        
        If `html` is True, will return the html output instead of rendering the visuals.
        '''

        if len(entities) <= 0:
            return displacy.render({'text':text, 'ents':[]}, manual=True, style="ent", jupyter= (not html))

        # huggingface NER format
        if isinstance(entities,list) and ('entity_group' in entities[0]):
            dic_ents = {
                "text": text,
                "ents": [
                    {"start": ent['start'], "end": ent['end'], "label": ent['entity_group']}
                        for ent in entities
                ],
                "title": None,
            }
        
        # AWS Comprehend detect
        elif isinstance(entities,list) and ('BeginOffset' in entities[0]):
            dic_ents = {
                "text": text,
                "ents": [
                    {"start": ent['BeginOffset'], "end": ent['EndOffset'], "label": ent.get('Type', '')}
                        for ent in entities
                ],
                "title": None,
            }

        # # Flair's NER format
        # elif isinstance(entities, Sentence):

        #     dic_ents = {
        #         "text": text,
        #         "ents": [
        #             {"start": ent.start_position, "end": ent.end_position, "label": ent.labels[0].value}
        #                 for ent in entities.get_spans('ner')
        #         ],
        #         "title": None,
        #     }

        return displacy.render(dic_ents, manual=True, style="ent", jupyter= (not html))
    
    @classmethod
    def disp_ds_ents(cls, ds:Dataset, idx:int, txt_src:str='content', html:bool=False):
        '''Display entities for the item in the dataset `ds`,
        located at index `idx`.
        Retrieves the text from the `txt_src` col and
        entities from the f'ents_{txt_src}' col.

        If `html` is True, will return the html output instead of rendering the visuals.
        '''
        col_ent = f'ents_{txt_src}'
        cols = [txt_src, col_ent]
        for col in cols:
            if col not in ds.column_names:
                raise ValueError(f'dataset `ds` MUST contain columns: {cols}')

        if isinstance(ds[0][col_ent][0], list): # old format: List[List[stt,end,ent_grp,score]]
            entities = [
                {
                    'start': int(stt),
                    'end': int(end),
                    'entity_group': MyNlp.ner_idx2tag[int(ent_grp)]
                }
                for (stt, end, ent_grp, score) in ds[idx][col_ent]
            ]
        elif isinstance(ds[0][col_ent][0], dict):
            entities = ds[idx][col_ent]

        return MyNlp.disp_ents(ds[idx][txt_src], entities, html)



# except ModuleNotFoundError:
#     print('[INFO] Cannot import `MyNlp` due to some package not found.. :) Sorry!')
