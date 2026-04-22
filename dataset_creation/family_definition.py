# Definizione delle famiglie di domande per il prompt schema, con mapping a campi di scena e program spec simbolici.

from typing import Any, Dict, List, Optional, Tuple

def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()

def build_param(name: str, p_type: str, optional: bool = False) -> Dict[str, Any]:
    return {
        "name": name,
        "type": p_type,
        "optional": bool(optional),
    }


def get_family_params_map() -> Dict[str, List[Dict[str, Any]]]:
    """
    Parametri tipizzati per ogni famiglia.
    Questi sono gli slot che il text template e il program template condividono,
    in stile CLEVR.
    """
    return {
        "main_source_identity": [],
        "secondary_source_identity": [],
        "background_source_identity": [],

        "source_presence_binary": [
            build_param("<SRC>", "SourceName"),
        ],
        "background_presence_binary": [],

        "source_count_estimation": [],
        "foreground_source_count": [],
        "background_source_count": [],

        "main_source_type": [],
        "main_source_role": [],
        "main_source_prominence": [],
        "main_source_timbre": [],

        "vocal_presence_binary": [],
        "vocal_type_classification": [],
        "instrumental_state_binary": [],

        "source_interaction_presence": [],
        "source_interaction_pair_identity": [],
        "source_interaction_overlap": [],
        "source_interaction_accompaniment": [],
        "background_vs_foreground_relation": [],

        "tempo_estimation": [],
        "rhythm_pattern_type": [],
        "arrangement_density": [],

        "recording_artifact_type": [],
        "environment_context_type": [],

        "audio_captioning": [],
        "conditioned_audio_captioning": [
            build_param("<COND_FOCUS>", "ConditionFocus"),
        ],
    }



def get_family_constraints_map() -> Dict[str, List[Dict[str, Any]]]:
    """
    Vincoli opzionali tra slot.
    Per ora minimi, ma già pronti per logica CLEVR-like.
    """
    return {
        "main_source_identity": [],
        "secondary_source_identity": [],
        "background_source_identity": [],
        "source_presence_binary": [],
        "background_presence_binary": [],
        "source_count_estimation": [],
        "foreground_source_count": [],
        "background_source_count": [],
        "main_source_type": [],
        "main_source_role": [],
        "main_source_prominence": [],
        "main_source_timbre": [],
        "vocal_presence_binary": [],
        "vocal_type_classification": [],
        "instrumental_state_binary": [],
        "source_interaction_presence": [],
        "source_interaction_pair_identity": [],
        "source_interaction_overlap": [],
        "source_interaction_accompaniment": [],
        "background_vs_foreground_relation": [],
        "tempo_estimation": [],
        "rhythm_pattern_type": [],
        "arrangement_density": [],
        "recording_artifact_type": [],
        "environment_context_type": [],
        "audio_captioning": [],
        "conditioned_audio_captioning": [],
    }


def get_family_scene_field_map() -> Dict[str, Dict[str, Any]]:
    """
    Mappa minima tra famiglia e contenuto di scena richiesto.
    Serve per fare family gating prima dell'instanziazione simbolica.
    """
    return {
        "main_source_identity": {
            "required_any": ["foreground_scene_objects", "scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "single_object_query",
        },
        "secondary_source_identity": {
            "required_any": ["scene_objects"],
            "min_counts": {"scene_objects": 2},
            "clevr_role": "single_object_query",
        },
        "background_source_identity": {
            "required_any": ["background_scene_objects", "explicit_background_relations"],
            "min_counts": {},
            "clevr_role": "single_object_query",
        },
        "source_presence_binary": {
            "required_any": ["scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "existence_query",
        },
        "background_presence_binary": {
            "required_any": ["background_scene_objects", "explicit_background_relations"],
            "min_counts": {},
            "clevr_role": "existence_query",
        },
        "source_count_estimation": {
            "required_any": ["scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "count_objects",
        },
        "foreground_source_count": {
            "required_any": ["foreground_scene_objects"],
            "min_counts": {"foreground_scene_objects": 1},
            "clevr_role": "count_objects",
        },
        "background_source_count": {
            "required_any": ["background_scene_objects", "explicit_background_relations"],
            "min_counts": {},
            "clevr_role": "count_objects",
        },
        "main_source_type": {
            "required_any": ["foreground_scene_objects", "scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "query_object_attribute",
        },
        "main_source_role": {
            "required_any": ["foreground_scene_objects", "scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "query_object_attribute",
        },
        "main_source_prominence": {
            "required_any": ["foreground_scene_objects", "scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "query_object_attribute",
        },
        "main_source_timbre": {
            "required_any": ["foreground_scene_objects", "scene_objects"],
            "min_counts": {"scene_objects": 1},
            "clevr_role": "single_object_description",
        },
        "vocal_presence_binary": {
            "required_any": ["has_vocals", "vocal_source_objects"],
            "min_counts": {},
            "clevr_role": "boolean_query",
        },
        "vocal_type_classification": {
            "required_any": ["vocal_source_objects"],
            "min_counts": {"vocal_source_objects": 1},
            "clevr_role": "query_object_identity",
        },
        "instrumental_state_binary": {
            "required_any": ["has_vocals", "explicitly_instrumental"],
            "min_counts": {},
            "clevr_role": "boolean_query",
        },
        "source_interaction_presence": {
            "required_any": ["supported_interaction_relation_triplets"],
            "min_counts": {"supported_interaction_relation_triplets": 1},
            "clevr_role": "existence_query",
        },
        "source_interaction_pair_identity": {
            "required_any": ["supported_interaction_relation_triplets"],
            "min_counts": {"supported_interaction_relation_triplets": 1},
            "clevr_role": "query_relation",
        },
        "source_interaction_overlap": {
            "required_any": ["supported_interaction_relation_triplets"],
            "min_counts": {"supported_interaction_relation_triplets": 1},
            "clevr_role": "query_relation",
        },
        "source_interaction_accompaniment": {
            "required_any": ["supported_interaction_relation_triplets"],
            "min_counts": {"supported_interaction_relation_triplets": 1},
            "clevr_role": "query_relation",
        },
        "background_vs_foreground_relation": {
            "required_any": ["explicit_background_relations", "background_scene_objects", "foreground_scene_objects"],
            "min_counts": {},
            "clevr_role": "query_relation",
        },
        "tempo_estimation": {
            "required_any": ["tempo_attribute_objects"],
            "min_counts": {"tempo_attribute_objects": 1},
            "clevr_role": "query_global_attribute",
        },
        "rhythm_pattern_type": {
            "required_any": ["rhythm_attribute_objects"],
            "min_counts": {"rhythm_attribute_objects": 1},
            "clevr_role": "query_global_attribute",
        },
        "arrangement_density": {
            "required_any": ["density_attribute_objects"],
            "min_counts": {"density_attribute_objects": 1},
            "clevr_role": "query_global_attribute",
        },
        "recording_artifact_type": {
            "required_any": ["quality_attribute_objects"],
            "min_counts": {"quality_attribute_objects": 1},
            "clevr_role": "query_global_attribute",
        },
        "environment_context_type": {
            "required_any": ["context_attribute_objects"],
            "min_counts": {"context_attribute_objects": 1},
            "clevr_role": "query_global_attribute",
        },
        "audio_captioning": {
            "required_any": ["scene_objects", "global_attribute_objects"],
            "min_counts": {},
            "clevr_role": "surface_realization_audio",
        },
        "conditioned_audio_captioning": {
            "required_any": ["scene_objects", "global_attribute_objects"],
            "min_counts": {},
            "clevr_role": "surface_realization_text_guided",
        },
    }


def get_default_program_spec_for_family(program_type: str) -> List[Dict[str, Any]]:
    """
    Functional program template simbolico di default per ciascuna famiglia finale.

    Importante:
    - questi sono template, non ancora esecuzione concreta
    - i campi 'value' con placeholder tipo <SRC> verranno istanziati dopo
    - la forma è più vicina alla logica CLEVR: programma + slot condivisi col testo
    """
    specs: Dict[str, List[Dict[str, Any]]] = {
        # ------------------------------------------------------------------
        # Gruppo 1 — Object / source existence and identity
        # ------------------------------------------------------------------
        "main_source_identity": [
            {"op": "select_objects", "source": "foreground_scene_objects", "fallback_source": "scene_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_identity", "object": "obj", "save_as": "answer"},
        ],
        "secondary_source_identity": [
            {"op": "select_objects", "source": "scene_objects", "save_as": "objs"},
            {"op": "sort_objects_by_salience", "objects": "objs", "save_as": "sorted_objs"},
            {"op": "pick_by_rank", "from": "sorted_objs", "rank": 2, "save_as": "obj"},
            {"op": "query_object_identity", "object": "obj", "save_as": "answer"},
        ],
        "background_source_identity": [
            {"op": "select_objects", "source": "background_scene_objects", "fallback_source": "background_objects_from_relations", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_identity", "object": "obj", "save_as": "answer"},
        ],
        "source_presence_binary": [
            {"op": "select_objects", "source": "scene_objects", "save_as": "objs"},
            {"op": "exists_object_with_identity", "objects": "objs", "value": "<SRC>", "save_as": "answer"},
        ],
        "background_presence_binary": [
            {"op": "query_scene_boolean_any", "fields": ["background_scene_objects", "explicit_background_relations"], "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 2 — Counting
        # ------------------------------------------------------------------
        "source_count_estimation": [
            {"op": "select_objects", "source": "scene_objects", "save_as": "objs"},
            {"op": "count_objects", "objects": "objs", "save_as": "answer"},
        ],
        "foreground_source_count": [
            {"op": "select_objects", "source": "foreground_scene_objects", "save_as": "objs"},
            {"op": "count_objects", "objects": "objs", "save_as": "answer"},
        ],
        "background_source_count": [
            {"op": "select_objects", "source": "background_scene_objects", "save_as": "objs"},
            {"op": "count_objects", "objects": "objs", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 3 — Source attributes
        # ------------------------------------------------------------------
        "main_source_type": [
            {"op": "select_objects", "source": "foreground_scene_objects", "fallback_source": "scene_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_attribute", "object": "obj", "attribute": "source_type", "save_as": "answer"},
        ],
        "main_source_role": [
            {"op": "select_objects", "source": "foreground_scene_objects", "fallback_source": "scene_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_attribute", "object": "obj", "attribute": "activity", "save_as": "answer"},
        ],
        "main_source_prominence": [
            {"op": "select_objects", "source": "foreground_scene_objects", "fallback_source": "scene_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_attribute", "object": "obj", "attribute": "prominence", "save_as": "answer"},
        ],
        "main_source_timbre": [
            {"op": "select_objects", "source": "foreground_scene_objects", "fallback_source": "scene_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_attribute", "object": "obj", "attribute": "timbre_label", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 4 — Vocal state
        # ------------------------------------------------------------------
        "vocal_presence_binary": [
            {"op": "query_scene_boolean", "field": "has_vocals", "save_as": "answer"},
        ],
        "vocal_type_classification": [
            {"op": "select_objects", "source": "vocal_source_objects", "save_as": "objs"},
            {"op": "pick_most_salient", "from": "objs", "save_as": "obj"},
            {"op": "query_object_identity", "object": "obj", "save_as": "answer"},
        ],
        "instrumental_state_binary": [
            {"op": "query_scene_boolean", "field": "explicitly_instrumental", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 5 — Pairwise source relations
        # ------------------------------------------------------------------
        "source_interaction_presence": [
            {"op": "query_scene_boolean", "field": "supported_interaction_relation_triplets", "save_as": "answer"},
        ],
        "source_interaction_pair_identity": [
            {"op": "select_relation_triplets", "source": "supported_interaction_relation_triplets", "save_as": "rels"},
            {"op": "pick_primary_relation", "from": "rels", "save_as": "rel"},
            {"op": "query_relation_pair_identity", "relation": "rel", "save_as": "answer"},
        ],
        "source_interaction_overlap": [
            {"op": "select_relation_triplets", "source": "supported_interaction_relation_triplets", "save_as": "rels"},
            {"op": "pick_primary_relation", "from": "rels", "save_as": "rel"},
            {"op": "relation_matches_type", "relation": "rel", "target_type": "overlaps_with", "save_as": "answer"},
        ],
        "source_interaction_accompaniment": [
            {"op": "select_relation_triplets", "source": "supported_interaction_relation_triplets", "save_as": "rels"},
            {"op": "pick_primary_relation", "from": "rels", "save_as": "rel"},
            {"op": "relation_in_type_set", "relation": "rel", "target_types": ["accompanies", "supports"], "save_as": "answer"},
        ],
        "background_vs_foreground_relation": [
            {"op": "select_relation_triplets", "source": "explicit_background_relation_triplets", "save_as": "rels"},
            {"op": "pick_primary_relation", "from": "rels", "save_as": "rel"},
            {"op": "relation_matches_type", "relation": "rel", "target_type": "background_to", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 6 — Global temporal / structural attributes
        # ------------------------------------------------------------------
        "tempo_estimation": [
            {"op": "select_global_attributes", "source": "tempo_attribute_objects", "save_as": "attrs"},
            {"op": "pick_primary_attribute", "from": "attrs", "save_as": "attr"},
            {"op": "query_global_attribute", "attribute_object": "attr", "save_as": "answer"},
        ],
        "rhythm_pattern_type": [
            {"op": "select_global_attributes", "source": "rhythm_attribute_objects", "save_as": "attrs"},
            {"op": "pick_primary_attribute", "from": "attrs", "save_as": "attr"},
            {"op": "query_global_attribute", "attribute_object": "attr", "save_as": "answer"},
        ],
        "arrangement_density": [
            {"op": "select_global_attributes", "source": "density_attribute_objects", "save_as": "attrs"},
            {"op": "pick_primary_attribute", "from": "attrs", "save_as": "attr"},
            {"op": "query_global_attribute", "attribute_object": "attr", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Gruppo 7 — Global recording / context
        # ------------------------------------------------------------------
        "recording_artifact_type": [
            {"op": "select_global_attributes", "source": "quality_attribute_objects", "save_as": "attrs"},
            {"op": "pick_primary_attribute", "from": "attrs", "save_as": "attr"},
            {"op": "query_global_attribute", "attribute_object": "attr", "save_as": "answer"},
        ],
        "environment_context_type": [
            {"op": "select_global_attributes", "source": "context_attribute_objects", "save_as": "attrs"},
            {"op": "pick_primary_attribute", "from": "attrs", "save_as": "attr"},
            {"op": "query_global_attribute", "attribute_object": "attr", "save_as": "answer"},
        ],

        # ------------------------------------------------------------------
        # Surface realization
        # ------------------------------------------------------------------
        "audio_captioning": [
            {"op": "surface_realization", "mode": "audio_driven", "save_as": "answer"},
        ],
        "conditioned_audio_captioning": [
            {"op": "surface_realization", "mode": "text_guided", "condition_focus": "<COND_FOCUS>", "save_as": "answer"},
        ],
    }
    return specs.get(normalize_text(program_type), [])

def get_family_fallback_templates(schema_item: Dict[str, Any]) -> List[str]:
    q_type = normalize_text(schema_item.get("question_type"))

    fallback_map = {
        "main_source_identity": [
            "Which sound source is the main one in this clip?",
            "What is the primary audible source in this clip?",
        ],
        "secondary_source_identity": [
            "What is the second most salient source in this clip?",
            "Which source is secondary to the main one in this clip?",
        ],
        "background_source_identity": [
            "Which sound source is present in the background of this clip?",
            "What background source can be heard behind the main source in this clip?",
        ],
        "source_presence_binary": [
            "Is <SRC> present in this clip?",
            "Can <SRC> be heard in this clip?",
        ],
        "background_presence_binary": [
            "Is there at least one background source in this clip?",
            "Can any background source be heard in this clip?",
        ],
        "source_count_estimation": [
            "How many salient sound sources are audible in this clip?",
            "How many distinct salient sources can be heard in this clip?",
        ],
        "foreground_source_count": [
            "How many foreground sources are present in this clip?",
            "How many prominent foreground sources can be heard in this clip?",
        ],
        "background_source_count": [
            "How many background sources are present in this clip?",
            "How many secondary background elements can be heard in this clip?",
        ],
        "main_source_type": [
            "What type of source is the main sound in this clip?",
            "Is the main source in this clip an instrument, a voice, or another source type?",
        ],
        "main_source_role": [
            "What functional role does the main source play in this clip?",
            "Is the main source acting as lead, accompaniment, rhythmic support, sustained background, intermittent support, or another role in this clip?",
        ],
        "main_source_prominence": [
            "What is the prominence level of the main source in this clip?",
            "Is the main source foreground, co-foreground, midground, or background in this clip?",
        ],
        "main_source_timbre": [
            "Which timbral or textural label best describes the main source in this clip?",
            "How would you characterize the timbre or texture of the main source in this clip?",
        ],
        "vocal_presence_binary": [
            "Are human voices present in this clip?",
            "Can any human voice be heard in this clip?",
        ],
        "vocal_type_classification": [
            "What type of voice is present in this clip?",
            "Which vocal type best describes the voice heard in this clip?",
        ],
        "instrumental_state_binary": [
            "Is this clip instrumental?",
            "Can this clip be described as instrumental rather than vocal?",
        ],
        "source_interaction_presence": [
            "Is there a supported relation between at least two sources in this clip?",
            "Can a clear interaction between sources be identified in this clip?",
        ],
        "source_interaction_pair_identity": [
            "Which two sources form the main interaction in this clip?",
            "What pair of sources is involved in the primary relation in this clip?",
        ],
        "source_interaction_overlap": [
            "Do the two main related sources overlap in this clip?",
            "Are the two main interacting sources sounding simultaneously in this clip?",
        ],
        "source_interaction_accompaniment": [
            "Does one source accompany or support another in this clip?",
            "Is there an accompaniment relation between the main related sources in this clip?",
        ],
        "background_vs_foreground_relation": [
            "Is the background source subordinated to the foreground source in this clip?",
            "Does the background source remain behind the foreground source in this clip?",
        ],
        "tempo_estimation": [
            "What is the tempo of this clip?",
            "How would you describe the tempo of this clip?",
        ],
        "rhythm_pattern_type": [
            "What is the rhythmic pattern of this clip?",
            "How would you describe the rhythmic profile of this clip?",
        ],
        "arrangement_density": [
            "What is the arrangement density of this clip?",
            "How dense or sparse is the arrangement in this clip?",
        ],
        "recording_artifact_type": [
            "Which recording-quality or artifact trait is most evident in this clip?",
            "What recording artifact or quality label best fits this clip?",
        ],
        "environment_context_type": [
            "Which environment context is most plausible for this clip?",
            "What context label best describes the environment suggested by this clip?",
        ],
        "audio_captioning": [
            "What are you hearing in this audio clip?",
            "Describe this audio clip.",
        ],
        "conditioned_audio_captioning": [
            "Describe this audio clip focusing on <COND_FOCUS>.",
            "What can be heard in this clip with attention to <COND_FOCUS>?",
        ],
    }

    default_text = schema_item.get("text", [""])
    return fallback_map.get(q_type, [default_text[0] if default_text else ""])

def build_prompt_schema() -> List[Dict[str, Any]]:
    """
    27 famiglie finali:
    - 25 famiglie CLEVR-like atomiche
    - 2 famiglie di surface realization

    Qui fissiamo:
    - text templates finali
    - params tipizzati
    - constraints
    - program templates

    Questo file NON implementa ancora l'executor.
    """
    params_map = get_family_params_map()
    constraints_map = get_family_constraints_map()

    family_defs: List[Dict[str, Any]] = [
        {
            "question_family_index": 1,
            "question_type": "main_source_identity",
            "slot_name": "main_source_identity",
            "answer_type": "source_name",
            "expected_focus": "main_source_identity",
            "difficulty": "easy",
            "diagnostic_role": "single_source_identification",
            "priority_tier": 1,
            "family_group": "source_existence_identity",
            "text": [
                "Which sound source is the main one in this clip?",
                "What is the primary audible source in this clip?",
                "Which source stands out the most in this clip?",
            ],
        },
        {
            "question_family_index": 2,
            "question_type": "secondary_source_identity",
            "slot_name": "secondary_source_identity",
            "answer_type": "source_name",
            "expected_focus": "secondary_source_identity",
            "difficulty": "medium",
            "diagnostic_role": "secondary_source_identification",
            "priority_tier": 2,
            "family_group": "source_existence_identity",
            "text": [
                "What is the second most salient source in this clip?",
                "Which source is secondary to the main one in this clip?",
            ],
        },
        {
            "question_family_index": 3,
            "question_type": "background_source_identity",
            "slot_name": "background_source_identity",
            "answer_type": "source_name",
            "expected_focus": "background_source_identity",
            "difficulty": "medium",
            "diagnostic_role": "background_source_identification",
            "priority_tier": 2,
            "family_group": "source_existence_identity",
            "text": [
                "Which sound source is present in the background of this clip?",
                "What background source can be heard behind the main source in this clip?",
            ],
        },
        {
            "question_family_index": 4,
            "question_type": "source_presence_binary",
            "slot_name": "source_presence_binary",
            "answer_type": "boolean",
            "expected_focus": "source_presence_binary",
            "difficulty": "easy",
            "diagnostic_role": "selected_source_presence",
            "priority_tier": 1,
            "family_group": "source_existence_identity",
            "text": [
                "Is <SRC> present in this clip?",
                "Can <SRC> be heard in this clip?",
                "Is there any audible <SRC> in this clip?",
            ],
        },
        {
            "question_family_index": 5,
            "question_type": "background_presence_binary",
            "slot_name": "background_presence_binary",
            "answer_type": "boolean",
            "expected_focus": "background_presence_binary",
            "difficulty": "easy",
            "diagnostic_role": "background_existence",
            "priority_tier": 1,
            "family_group": "source_existence_identity",
            "text": [
                "Is there at least one background source in this clip?",
                "Can any background source be heard in this clip?",
            ],
        },
        {
            "question_family_index": 6,
            "question_type": "source_count_estimation",
            "slot_name": "source_count_estimation",
            "answer_type": "count",
            "expected_focus": "source_count_estimation",
            "difficulty": "easy",
            "diagnostic_role": "count_salient_sources",
            "priority_tier": 1,
            "family_group": "counting",
            "text": [
                "How many salient sound sources are audible in this clip?",
                "How many distinct salient sources can be heard in this clip?",
            ],
        },
        {
            "question_family_index": 7,
            "question_type": "foreground_source_count",
            "slot_name": "foreground_source_count",
            "answer_type": "count",
            "expected_focus": "foreground_source_count",
            "difficulty": "easy",
            "diagnostic_role": "count_foreground_sources",
            "priority_tier": 1,
            "family_group": "counting",
            "text": [
                "How many foreground sources are present in this clip?",
                "How many prominent foreground sources can be heard in this clip?",
            ],
        },
        {
            "question_family_index": 8,
            "question_type": "background_source_count",
            "slot_name": "background_source_count",
            "answer_type": "count",
            "expected_focus": "background_source_count",
            "difficulty": "medium",
            "diagnostic_role": "count_background_sources",
            "priority_tier": 2,
            "family_group": "counting",
            "text": [
                "How many background sources are present in this clip?",
                "How many secondary background elements can be heard in this clip?",
            ],
        },
        {
            "question_family_index": 9,
            "question_type": "main_source_type",
            "slot_name": "main_source_type",
            "answer_type": "source_type",
            "expected_focus": "main_source_type",
            "difficulty": "easy",
            "diagnostic_role": "query_main_source_type",
            "priority_tier": 1,
            "family_group": "source_attributes",
            "text": [
                "What type of source is the main sound in this clip?",
                "Is the main source in this clip an instrument, a voice, or another source type?",
            ],
        },
        {
            "question_family_index": 10,
            "question_type": "main_source_role",
            "slot_name": "main_source_role",
            "answer_type": "activity",
            "expected_focus": "main_source_role",
            "difficulty": "medium",
            "diagnostic_role": "query_main_source_role",
            "priority_tier": 1,
            "family_group": "source_attributes",
            "text": [
                "What functional role does the main source play in this clip?",
                "Is the main source acting as lead, accompaniment, rhythmic support, sustained background, intermittent support, or another role in this clip?",
            ],
        },
        {
            "question_family_index": 11,
            "question_type": "main_source_prominence",
            "slot_name": "main_source_prominence",
            "answer_type": "prominence",
            "expected_focus": "main_source_prominence",
            "difficulty": "easy",
            "diagnostic_role": "query_main_source_prominence",
            "priority_tier": 2,
            "family_group": "source_attributes",
            "text": [
                "What is the prominence level of the main source in this clip?",
                "Is the main source foreground, co-foreground, midground, or background in this clip?",
            ],
        },
        {
            "question_family_index": 12,
            "question_type": "main_source_timbre",
            "slot_name": "main_source_timbre",
            "answer_type": "timbre_label",
            "expected_focus": "main_source_timbre",
            "difficulty": "medium",
            "diagnostic_role": "query_main_source_timbre",
            "priority_tier": 2,
            "family_group": "source_attributes",
            "text": [
                "Which timbral or textural label best describes the main source in this clip?",
                "How would you characterize the timbre or texture of the main source in this clip?",
            ],
        },
        {
            "question_family_index": 13,
            "question_type": "vocal_presence_binary",
            "slot_name": "vocal_presence_binary",
            "answer_type": "boolean",
            "expected_focus": "vocal_presence_binary",
            "difficulty": "easy",
            "diagnostic_role": "voice_presence",
            "priority_tier": 1,
            "family_group": "vocal_state",
            "text": [
                "Are human voices present in this clip?",
                "Can any human voice be heard in this clip?",
            ],
        },
        {
            "question_family_index": 14,
            "question_type": "vocal_type_classification",
            "slot_name": "vocal_type_classification",
            "answer_type": "voice_identity",
            "expected_focus": "vocal_type_classification",
            "difficulty": "medium",
            "diagnostic_role": "voice_type_query",
            "priority_tier": 2,
            "family_group": "vocal_state",
            "text": [
                "What type of voice is present in this clip?",
                "Which vocal type best describes the voice heard in this clip?",
            ],
        },
        {
            "question_family_index": 15,
            "question_type": "instrumental_state_binary",
            "slot_name": "instrumental_state_binary",
            "answer_type": "boolean",
            "expected_focus": "instrumental_state_binary",
            "difficulty": "easy",
            "diagnostic_role": "instrumental_state_query",
            "priority_tier": 1,
            "family_group": "vocal_state",
            "text": [
                "Is this clip instrumental?",
                "Can this clip be described as instrumental rather than vocal?",
            ],
        },
        {
            "question_family_index": 16,
            "question_type": "source_interaction_presence",
            "slot_name": "source_interaction_presence",
            "answer_type": "boolean",
            "expected_focus": "source_interaction_presence",
            "difficulty": "easy",
            "diagnostic_role": "relation_existence",
            "priority_tier": 1,
            "family_group": "pairwise_relations",
            "text": [
                "Is there a supported relation between at least two sources in this clip?",
                "Can a clear interaction between sources be identified in this clip?",
            ],
        },
        {
            "question_family_index": 17,
            "question_type": "source_interaction_pair_identity",
            "slot_name": "source_interaction_pair_identity",
            "answer_type": "source_pair",
            "expected_focus": "source_interaction_pair_identity",
            "difficulty": "medium",
            "diagnostic_role": "main_relation_pair_identity",
            "priority_tier": 2,
            "family_group": "pairwise_relations",
            "text": [
                "Which two sources form the main interaction in this clip?",
                "What pair of sources is involved in the primary relation in this clip?",
            ],
        },
        {
            "question_family_index": 18,
            "question_type": "source_interaction_overlap",
            "slot_name": "source_interaction_overlap",
            "answer_type": "boolean",
            "expected_focus": "source_interaction_overlap",
            "difficulty": "medium",
            "diagnostic_role": "overlap_relation_query",
            "priority_tier": 2,
            "family_group": "pairwise_relations",
            "text": [
                "Do the two main related sources overlap in this clip?",
                "Are the two main interacting sources sounding simultaneously in this clip?",
            ],
        },
        {
            "question_family_index": 19,
            "question_type": "source_interaction_accompaniment",
            "slot_name": "source_interaction_accompaniment",
            "answer_type": "boolean",
            "expected_focus": "source_interaction_accompaniment",
            "difficulty": "medium",
            "diagnostic_role": "accompaniment_relation_query",
            "priority_tier": 2,
            "family_group": "pairwise_relations",
            "text": [
                "Does one source accompany or support another in this clip?",
                "Is there an accompaniment relation between the main related sources in this clip?",
            ],
        },
        {
            "question_family_index": 20,
            "question_type": "background_vs_foreground_relation",
            "slot_name": "background_vs_foreground_relation",
            "answer_type": "boolean",
            "expected_focus": "background_vs_foreground_relation",
            "difficulty": "medium",
            "diagnostic_role": "background_foreground_relation_query",
            "priority_tier": 2,
            "family_group": "pairwise_relations",
            "text": [
                "Is the background source subordinated to the foreground source in this clip?",
                "Does the background source remain behind the foreground source in this clip?",
            ],
        },
        {
            "question_family_index": 21,
            "question_type": "tempo_estimation",
            "slot_name": "tempo_estimation",
            "answer_type": "tempo_term",
            "expected_focus": "tempo_estimation",
            "difficulty": "easy",
            "diagnostic_role": "tempo_query",
            "priority_tier": 1,
            "family_group": "global_temporal_structural",
            "text": [
                "What is the tempo of this clip?",
                "How would you describe the tempo of this clip?",
            ],
        },
        {
            "question_family_index": 22,
            "question_type": "rhythm_pattern_type",
            "slot_name": "rhythm_pattern_type",
            "answer_type": "rhythm_term",
            "expected_focus": "rhythm_pattern_type",
            "difficulty": "easy",
            "diagnostic_role": "rhythm_query",
            "priority_tier": 1,
            "family_group": "global_temporal_structural",
            "text": [
                "What is the rhythmic pattern of this clip?",
                "How would you describe the rhythmic profile of this clip?",
            ],
        },
        {
            "question_family_index": 23,
            "question_type": "arrangement_density",
            "slot_name": "arrangement_density",
            "answer_type": "density_term",
            "expected_focus": "arrangement_density",
            "difficulty": "easy",
            "diagnostic_role": "density_query",
            "priority_tier": 1,
            "family_group": "global_temporal_structural",
            "text": [
                "What is the arrangement density of this clip?",
                "How dense or sparse is the arrangement in this clip?",
            ],
        },
        {
            "question_family_index": 24,
            "question_type": "recording_artifact_type",
            "slot_name": "recording_artifact_type",
            "answer_type": "recording_quality_term",
            "expected_focus": "recording_artifact_type",
            "difficulty": "medium",
            "diagnostic_role": "recording_quality_query",
            "priority_tier": 2,
            "family_group": "global_recording_context",
            "text": [
                "Which recording-quality or artifact trait is most evident in this clip?",
                "What recording artifact or quality label best fits this clip?",
            ],
        },
        {
            "question_family_index": 25,
            "question_type": "environment_context_type",
            "slot_name": "environment_context_type",
            "answer_type": "context_term",
            "expected_focus": "environment_context_type",
            "difficulty": "medium",
            "diagnostic_role": "context_query",
            "priority_tier": 2,
            "family_group": "global_recording_context",
            "text": [
                "Which environment context is most plausible for this clip?",
                "What context label best describes the environment suggested by this clip?",
            ],
        },
        {
            "question_family_index": 26,
            "question_type": "audio_captioning",
            "slot_name": "audio_captioning",
            "answer_type": "free_caption",
            "expected_focus": "audio_driven_surface_realization",
            "difficulty": "medium",
            "diagnostic_role": "audio_only_captioning",
            "priority_tier": 3,
            "family_group": "surface_realization",
            "text": [
                "What are you hearing in this audio clip?",
                "Describe this audio clip.",
            ],
        },
        {
            "question_family_index": 27,
            "question_type": "conditioned_audio_captioning",
            "slot_name": "conditioned_audio_captioning",
            "answer_type": "free_caption",
            "expected_focus": "text_guided_surface_realization",
            "difficulty": "medium",
            "diagnostic_role": "text_conditioned_captioning",
            "priority_tier": 3,
            "family_group": "surface_realization",
            "text": [
                "Describe this audio clip focusing on <COND_FOCUS>.",
                "What can be heard in this clip with attention to <COND_FOCUS>?",
                "Provide a short description of this clip focusing on <COND_FOCUS>.",
            ],
        },
    ]

    for item in family_defs:
        q_type = item["question_type"]
        item["params"] = params_map.get(q_type, [])
        item["constraints"] = constraints_map.get(q_type, [])
        item["program_type"] = q_type
        item["program"] = get_default_program_spec_for_family(q_type)
        item["fallback_text"] = get_family_fallback_templates(item)

    return family_defs