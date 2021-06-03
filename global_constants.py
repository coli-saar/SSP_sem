import os


class CONST:
    """
    constants that does not change during the
            *****PROJECT*****
    for execution settings, see specific scripts
    """
    '== Paths ==================================='
    data_folder = os.path.join('script_parser', 'clean_data_100011111')
    toy_data_folder = os.path.join('script_parser', 'toy')

    '== PreProcessing ==========================='
    null_label = 'none'

    merge_irregular_labels = True
    irregular_prefixes = \
        {'events': ['unrelev_', 'relnscrev_', 'screv_other', 'unclear_', 'irregular'],
         'participants': ['npart_', 'no_label', 'scrpart_other', 'suppvcomp', 'unclear']}
    irregular_event_label = '#irregularE'
    irregular_participant_label = '@irregularP'
    regular_event_label = '#regularE'
    regular_participant_label = '@regularP'

    event_label_prefix = '#'
    participant_label_prefix = '@'
    transformer_dummy_tag = 'X'

    begin_of_story_event = '<story_begins>'
    begin_of_story_type = '<bost>'
    end_of_story_type = '<end_of_story>'

    '==  Stats ================================='
    scenario_s = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
    effective_scenario_s = ['bath']

    #
    begin_of_sequence = '<bose>'
    end_of_sequence = '<eose>'
    dummy_event_annotation_prefix = '<no_event_annotated>'
