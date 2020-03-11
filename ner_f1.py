import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from copy import deepcopy

from collections import namedtuple

# Evaluation metric for Innoplexus NER Challenge

def collect_named_entities(tokens): # Helper Function for score calculation
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """
    Entity = namedtuple("Entity", "e_type start_offset end_offset")
    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities

def compute_metrics(true_named_entities, pred_named_entities): # Helper Function for score calculation
    eval_metrics = {'correct': 0, 'partial': 0, 'missed': 0, 'spurius': 0}
    target_tags_no_schema = ['indications']

    # overall results
    evaluation = {'partial': deepcopy(eval_metrics)}


    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # check if there's an exact match, i.e.: boundary and entity type match
        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities
            for true in true_named_entities:

                
                # 2. check for an overlap i.e. not exact boundary match, with true entities
                if pred.start_offset <= true.end_offset and true.start_offset <= pred.end_offset:

                    true_which_overlapped_with_pred.append(true)

                    evaluation['partial']['partial'] += 1

                    found_overlap = True
                    break

            # count spurius (i.e., False Positive) entities
            if not found_overlap:
                # overall results
                evaluation['partial']['spurius'] += 1

    # count missed entities (i.e. False Negative)
    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['partial']['missed'] += 1

    # Compute 'possible', 'actual'
    for eval_type in ['partial']:

        correct = evaluation[eval_type]['correct']
        partial = evaluation[eval_type]['partial']
        missed = evaluation[eval_type]['missed']
        spurius = evaluation[eval_type]['spurius']

        # possible: nr. annotations in the gold-standard which contribute to the final score
        evaluation[eval_type]['possible'] = correct + partial + missed

        # actual: number of annotations produced by the NER system
        evaluation[eval_type]['actual'] = correct + partial + spurius

        actual = evaluation[eval_type]['actual']
        possible = evaluation[eval_type]['possible']

    return evaluation

def list_converter(df): # Helper Function for score calculation
    keys, values = df.sort_values('Sent_ID_x').values.T
    ukeys, index = np.unique(keys,True)
    lists = [list(array) for array in np.split(values,index[1:])]
    return lists

# ideal and pred respectively represent dataframes containing actual labels and predictions for the set of sentences in the test data. 
# It has the same format as the sample submission (id, Sent_ID, tag)

def calculate_score(ideal, pred): # Calculates the final F1 Score

    merged = ideal.merge(pred, on = "id", how="inner").drop(['Sent_ID_y'],axis = 1)
    
    
    # The scores are calculated sentence wise and then aggregated to calculate the overall score, for this
    # List converter function groups the labels by sentence to generate a list of lists with each inner list representing a sentence in sequence
    ideal_ = list_converter(merged.drop(['id','tag_y'],axis = 1))
    pred_ = list_converter(merged.drop(['id','tag_x'],axis = 1))

    metrics_results = {'correct': 0, 'partial': 0,
                   'missed': 0, 'spurius': 0, 'possible': 0, 'actual': 0}

    results = {'partial': deepcopy(metrics_results)}


    for true_ents, pred_ents in zip(ideal_, pred_):    
    # compute results for one sentence
        tmp_results = compute_metrics(collect_named_entities(true_ents),collect_named_entities(pred_ents))
    
    # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]
    correct = results['partial']['correct']
    partial = results['partial']['partial']
    missed = results['partial']['missed']
    spurius = results['partial']['spurius']
    actual = results['partial']['actual']
    possible = results['partial']['possible']


    precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
    recall = (correct + 0.5 * partial) / possible if possible > 0 else 0


    score = (2 * precision * recall)/(precision + recall) if (precision + recall) >0 else 0
    
    # final score
    return score