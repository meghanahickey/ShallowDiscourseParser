#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The Official CONLL 2016 Shared Task Scorer

"""
import argparse
import json

from confusion_matrix import ConfusionMatrix, Alphabet
import validator


def evaluate(gold_list, predicted_list):
    sense_cm = evaluate_sense(gold_list, predicted_list)

    print('Sense classification--------------')
    sense_cm.print_summary()
    print('Overall parser performance --------------')
    precision, recall, f1 = sense_cm.compute_micro_average_f1()
    print('Precision %1.4f Recall %1.4f F1 %1.4f' % (precision, recall, f1))
    #return connective_cm, arg1_cm, arg2_cm, rel_arg_cm, sense_cm, precision, recall, f1
    return sense_cm, precision, recall, f1

def spans_exact_matching(gold_doc_id_spans, predicted_doc_id_spans):
    """Matching two lists of spans

    Input:
        gold_doc_id_spans : (DocID , a list of lists of tuples of token addresses)
        predicted_doc_id_spans : (DocID , a list of lists of token indices)

    Returns:
        True if the spans match exactly
    """
    exact_match = True
    gold_docID = gold_doc_id_spans[0]
    gold_spans = gold_doc_id_spans[1]
    predicted_docID = predicted_doc_id_spans[0]
    predicted_spans = predicted_doc_id_spans[1]

    for gold_span, predicted_span in zip(gold_spans, predicted_spans):
        exact_match = span_exact_matching((gold_docID,gold_span), (predicted_docID, predicted_span)) \
                and exact_match
    return exact_match

def span_exact_matching(gold_span, predicted_span):
    """Matching two spans

    Input:
        gold_span : a list of tuples :(DocID, list of tuples of token addresses)
        predicted_span : a list of tuples :(DocID, list of token indices)

    Returns:
        True if the spans match exactly
    """
    gold_docID = gold_span[0]
    predicted_docID = predicted_span[0]
    if gold_docID != predicted_docID:
        return False
    gold_token_indices = [x[2] for x in gold_span[1]]
    predicted_token_indices = predicted_span[1]
    return gold_docID == predicted_docID and gold_token_indices == predicted_token_indices

def evaluate_sense(gold_list, predicted_list):
    """Evaluate sense classifier

    The label ConfusionMatrix.NEGATIVE_CLASS is for the relations 
    that are missed by the system
    because the arguments don't match any of the gold relations.
    """
    sense_alphabet = Alphabet()
    valid_senses = validator.identify_valid_senses(gold_list)
    for relation in gold_list:
        sense = relation['Sense'][0]
        if sense in valid_senses:
            sense_alphabet.add(sense)

    sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)

    sense_cm = ConfusionMatrix(sense_alphabet)
    gold_to_predicted_map, predicted_to_gold_map = \
            _link_gold_predicted(gold_list, predicted_list, spans_exact_matching)

    for i, gold_relation in enumerate(gold_list):
        gold_sense = gold_relation['Sense'][0]
        if gold_sense in valid_senses:
            if i in gold_to_predicted_map:
                predicted_sense = gold_to_predicted_map[i]['Sense'][0]
                if predicted_sense in gold_relation['Sense']:
                    sense_cm.add(predicted_sense, predicted_sense)
                else:
                    if not sense_cm.alphabet.has_label(predicted_sense):
                        predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
                    sense_cm.add(predicted_sense, gold_sense)
            else:
                sense_cm.add(ConfusionMatrix.NEGATIVE_CLASS, gold_sense)

    for i, predicted_relation in enumerate(predicted_list):
        if i not in predicted_to_gold_map:
            predicted_sense = predicted_relation['Sense'][0]
            if not sense_cm.alphabet.has_label(predicted_sense):
                predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
            sense_cm.add(predicted_sense, ConfusionMatrix.NEGATIVE_CLASS)
    return sense_cm

def _link_gold_predicted(gold_list, predicted_list, matching_fn):
    """Link gold standard relations to the predicted relations

    A pair of relations are linked when the arg1 and the arg2 match exactly.
    We do this because we want to evaluate sense classification later.

    Returns:
        A tuple of two dictionaries:
        1) mapping from gold relation index to predicted relation index
        2) mapping from predicted relation index to gold relation index
    """
    gold_to_predicted_map = {}
    predicted_to_gold_map = {}
    gold_arg12_list = [(x['DocID'], (x['Arg1']['TokenList'], x['Arg2']['TokenList']))
            for x in gold_list]
    predicted_arg12_list = [(x['DocID'], (x['Arg1']['TokenList'], x['Arg2']['TokenList']))
            for x in predicted_list]
    for gi, gold_span in enumerate(gold_arg12_list):
        for pi, predicted_span in enumerate(predicted_arg12_list):
            if matching_fn(gold_span, predicted_span):
                gold_to_predicted_map[gi] = predicted_list[pi]
                predicted_to_gold_map[pi] = gold_list[gi]
    return gold_to_predicted_map, predicted_to_gold_map


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate system's output against the gold standard")
    parser.add_argument('gold', help='Gold standard file')
    parser.add_argument('predicted', help='System output file')
    args = parser.parse_args()
    gold_list = [json.loads(x) for x in open(args.gold)]
    predicted_list = [json.loads(x) for x in open(args.predicted)]
    print('\n================================================')
    print('Evaluation for all discourse relations')
    evaluate(gold_list, predicted_list)

    print('\n================================================')
    print('Evaluation for explicit discourse relations only')
    explicit_gold_list = [x for x in gold_list if x['Type'] == 'Explicit']
    explicit_predicted_list = [x for x in predicted_list if x['Type'] == 'Explicit']
    evaluate(explicit_gold_list, explicit_predicted_list)

    print('\n================================================')
    print('Evaluation for non-explicit discourse relations only (Implicit, EntRel, AltLex)')
    non_explicit_gold_list = [x for x in gold_list if x['Type'] != 'Explicit']
    non_explicit_predicted_list = [x for x in predicted_list if x['Type'] != 'Explicit']
    evaluate(non_explicit_gold_list, non_explicit_predicted_list)

if __name__ == '__main__':
    main()

