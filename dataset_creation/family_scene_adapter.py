from typing import Any, Dict, List, Optional, Tuple

def normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()


def lowercase_text(x: Optional[str]) -> str:
    return normalize_text(x).lower()


def dedupe_keep_order(items: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def normalize_support_level(x: Optional[str]) -> str:
    s = lowercase_text(x)
    if s in {"strong", "plausible", "weak"}:
        return s
    return "weak"


def support_rank(x: Optional[str]) -> int:
    s = normalize_support_level(x)
    if s == "strong":
        return 2
    if s == "plausible":
        return 1
    return 0


def sort_scene_objects_for_salience(scene_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prominence_score = {
        "foreground": 4,
        "co-foreground": 3,
        "midground": 2,
        "background": 1,
        "uncertain": 0,
    }

    def key_fn(obj: Dict[str, Any]):
        return (
            support_rank(obj.get("support_level")),
            prominence_score.get(obj.get("prominence", "uncertain"), 0),
            1 if obj.get("source_type") == "voice" else 0,
            lowercase_text(obj.get("canonical_name")),
        )

    return sorted(scene_objects, key=key_fn, reverse=True)


def build_attribute_object_views(global_attribute_objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    tempo = []
    rhythm = []
    density = []
    quality = []
    context = []

    for attr in global_attribute_objects or []:
        kind = normalize_text(attr.get("attribute_kind"))
        if kind == "tempo":
            tempo.append(attr)
        elif kind == "rhythm":
            rhythm.append(attr)
        elif kind == "density":
            density.append(attr)
        elif kind == "recording_quality":
            quality.append(attr)
        elif kind == "environment_context":
            context.append(attr)

    def sort_attrs(attrs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            attrs,
            key=lambda x: (support_rank(x.get("support_level")), lowercase_text(x.get("value"))),
            reverse=True,
        )

    return {
        "tempo_attribute_objects": sort_attrs(tempo),
        "rhythm_attribute_objects": sort_attrs(rhythm),
        "density_attribute_objects": sort_attrs(density),
        "quality_attribute_objects": sort_attrs(quality),
        "context_attribute_objects": sort_attrs(context),
    }


def build_relation_views(
    scene_objects: List[Dict[str, Any]],
    scene_relation_triplets: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    name_to_obj = {obj["canonical_name"]: obj for obj in scene_objects}
    supported_interaction_relation_triplets = []
    explicit_background_relation_triplets = []
    explicit_background_relations = []

    interaction_types = {
        "co_occurs_with",
        "overlaps_with",
        "accompanies",
        "supports",
        "alternates_with",
    }

    for rel_type, a_name, b_name in scene_relation_triplets or []:
        if rel_type in interaction_types:
            supported_interaction_relation_triplets.append((rel_type, a_name, b_name))
        if rel_type == "background_to":
            explicit_background_relation_triplets.append((rel_type, a_name, b_name))
            explicit_background_relations.append({
                "type": rel_type,
                "source_a_name": a_name,
                "source_b_name": b_name,
                "source_a_object": name_to_obj.get(a_name),
                "source_b_object": name_to_obj.get(b_name),
            })

    return {
        "supported_interaction_relation_triplets": dedupe_keep_order(supported_interaction_relation_triplets),
        "explicit_background_relation_triplets": dedupe_keep_order(explicit_background_relation_triplets),
        "explicit_background_relations": explicit_background_relations,
    }


def build_family_scene_view(projected_scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_objects = sort_scene_objects_for_salience(projected_scene.get("scene_objects", []))
    global_attribute_objects = projected_scene.get("global_attribute_objects", [])
    scene_relation_triplets = projected_scene.get("scene_relation_triplets", [])

    foreground_scene_objects = [
        obj for obj in scene_objects
        if obj.get("prominence") in {"foreground", "co-foreground"}
        and normalize_support_level(obj.get("support_level")) in {"strong", "plausible"}
    ]

    background_scene_objects = [
        obj for obj in scene_objects
        if obj.get("prominence") == "background"
        and normalize_support_level(obj.get("support_level")) in {"strong", "plausible"}
    ]

    vocal_source_objects = [
        obj for obj in scene_objects
        if obj.get("source_type") == "voice"
        and normalize_support_level(obj.get("support_level")) in {"strong", "plausible"}
    ]

    attr_views = build_attribute_object_views(global_attribute_objects)
    rel_views = build_relation_views(scene_objects, scene_relation_triplets)

    background_objects_from_relations = []
    for rel in rel_views.get("explicit_background_relations", []):
        obj = rel.get("source_a_object")
        if obj is not None:
            background_objects_from_relations.append(obj)

    background_objects_from_relations = dedupe_keep_order(background_objects_from_relations)

    out = {
        "audio_id": projected_scene.get("audio_id"),
        "caption": projected_scene.get("caption", ""),
        "aspect_list": projected_scene.get("aspect_list", []),

        "scene_objects": scene_objects,
        "foreground_scene_objects": foreground_scene_objects,
        "background_scene_objects": background_scene_objects,
        "vocal_source_objects": vocal_source_objects,

        "has_vocals": bool(projected_scene.get("has_vocals", False)),
        "explicitly_instrumental": bool(projected_scene.get("explicitly_instrumental", False)),

        "scene_relations": projected_scene.get("scene_relations", []),
        "scene_relation_triplets": scene_relation_triplets,

        **rel_views,
        **attr_views,

        "global_attribute_objects": global_attribute_objects,
        "source_strengths": projected_scene.get("source_strengths", {}),
        "background_objects_from_relations": background_objects_from_relations,
    }
    return out


def build_family_support_from_scene_field_map(
    family_scene_view: Dict[str, Any],
    family_scene_field_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    support_map: Dict[str, Dict[str, Any]] = {}

    for family_name, spec in (family_scene_field_map or {}).items():
        required_any = spec.get("required_any", []) or []
        min_counts = spec.get("min_counts", {}) or {}
        clevr_role = spec.get("clevr_role", "")

        available_any = False
        available_reasons = []

        for field_name in required_any:
            value = family_scene_view.get(field_name)

            if isinstance(value, list):
                if len(value) > 0:
                    available_any = True
                    available_reasons.append(f"{field_name}:list_nonempty")
            elif isinstance(value, bool):
                if value is True:
                    available_any = True
                    available_reasons.append(f"{field_name}:bool_true")
                elif field_name in family_scene_view:
                    if field_name in {"has_vocals", "explicitly_instrumental"}:
                        available_any = True
                        available_reasons.append(f"{field_name}:bool_present")
            elif value is not None:
                available_any = True
                available_reasons.append(f"{field_name}:present")

        min_counts_ok = True
        min_count_failures = []

        for field_name, min_required in min_counts.items():
            value = family_scene_view.get(field_name, [])
            actual = len(value) if isinstance(value, list) else 0
            if actual < int(min_required):
                min_counts_ok = False
                min_count_failures.append({
                    "field": field_name,
                    "required": int(min_required),
                    "actual": actual,
                })

        supported = bool(available_any and min_counts_ok)

        support_map[family_name] = {
            "supported": supported,
            "required_any": required_any,
            "min_counts": min_counts,
            "clevr_role": clevr_role,
            "available_any": available_any,
            "available_reasons": available_reasons,
            "min_counts_ok": min_counts_ok,
            "min_count_failures": min_count_failures,
        }

    return support_map