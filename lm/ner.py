from gliner import GLiNER


class NER:
    entity_labels = [
        "person",
        "organization",
        "location",
        "event",
        "date",
        "product",
        "law",
        "medical",
        "scientific_term",
        "work_of_art",
        "language",
        "nationality",
        "religion",
        "sport",
        "weapon",
        "food",
        "currency",
        "disease",
        "animal",
        "plant",
    ]

    def __init__(self):
        self.ner_model = GLiNER.from_pretrained("numind/NuNerZero")

    def extract_entities(self, text: str) -> dict[str, any]:
        """return entity format: 'thomas jeffrey hanks': 'person'"""

        # entity format: {'start': 2, 'end': 8, 'text': 'thomas', 'label': 'person', 'score': 0.9884123206138611}
        entities = self.ner_model.predict_entities(text, NER.entity_labels)

        # entity format: 'thomas jeffrey hanks': 'person'
        entities = self._merge_entities(text, entities)

        # turn entity text to lowercase
        entities = {entity.lower(): label for entity, label in entities.items()}

        return entities

    def _merge_entities(
        self, text: str, entities: list[dict[str, any]]
    ) -> dict[str, str]:
        """{'start': 2, 'end': 8, 'text': 'thomas', 'label': 'person', 'score': 0.9884123206138611}

        → ''thomas jeffrey hanks': 'person'"""
        if not entities:
            return {}

        # merge entities
        merged_entities = []
        for entity in entities:
            if len(merged_entities) > 0 and (
                entity["start"] - merged_entities[-1]["end"] <= 1
            ):
                merged_entities[-1]["end"] = entity["end"]
                merged_entities[-1]["text"] = text[
                    merged_entities[-1]["start"] : merged_entities[-1]["end"]
                ].strip()
            else:
                merged_entities.append(entity)

        # make entity label pairs
        entity_label_pairs = {
            entity["text"]: entity["label"] for entity in merged_entities
        }
        return entity_label_pairs
