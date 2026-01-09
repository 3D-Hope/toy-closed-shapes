THREED_FRONT_BEDROOM_FURNITURE = {
    "desk": "desk",
    "nightstand": "nightstand",
    "king-size bed": "double_bed",
    "single bed": "single_bed",
    "kids bed": "kids_bed",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "pendant_lamp",
    "bookcase/jewelry armoire": "bookshelf",
    "tv stand": "tv_stand",
    "wardrobe": "wardrobe",
    "lounge chair/cafe chair/office chair": "chair",
    "dining chair": "chair",
    "classic chinese chair": "chair",
    "armchair": "armchair",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
    "corner/side table": "table",
    "dining table": "table",
    "round end table": "table",
    "drawer chest/corner cabinet": "cabinet",
    "sideboard/side cabinet/console table": "cabinet",
    "children cabinet": "children_cabinet",
    "shelf": "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "coffee table": "coffee_table",
    "loveseat sofa": "sofa",
    "three-seat/multi-seat sofa": "sofa",
    "l-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "chaise longue sofa": "sofa",
}

THREED_FRONT_LIBRARY_FURNITURE = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "dressing table": "dressing_table",
    "dressing chair": "dressing_chair",
}

THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase/jewelry armoire": "bookshelf",
    "desk": "desk",
    "pendant lamp": "pendant_lamp",
    "ceiling lamp": "ceiling_lamp",
    "lounge chair/cafe chair/office chair": "lounge_chair",
    "dining chair": "dining_chair",
    "dining table": "dining_table",
    "corner/side table": "corner_side_table",
    "classic chinese chair": "chinese_chair",
    "armchair": "armchair",
    "shelf": "shelf",
    "sideboard/side cabinet/console table": "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool": "stool",
    "round end table": "round_end_table",
    "loveseat sofa": "loveseat_sofa",
    "drawer chest/corner cabinet": "cabinet",
    "wardrobe": "wardrobe",
    "three-seat/multi-seat sofa": "multi_seat_sofa",
    "wine cabinet": "wine_cabinet",
    "coffee table": "coffee_table",
    "lazy sofa": "lazy_sofa",
    "children cabinet": "cabinet",
    "chaise longue sofa": "chaise_longue_sofa",
    "l-shaped sofa": "l_shaped_sofa",
    "tv stand": "tv_stand",
}

# Convert the above dictionaries to json
import json

# Create JSONs where keys are indices and values are unique values from the original dicts


def make_unique_value_dict(d):
    unique_values = sorted(set(d.values()))
    value_dict = {i: v for i, v in enumerate(unique_values)}
    return value_dict


bedroom_unique = make_unique_value_dict(THREED_FRONT_BEDROOM_FURNITURE)
library_unique = make_unique_value_dict(THREED_FRONT_LIBRARY_FURNITURE)
livingroom_unique = make_unique_value_dict(THREED_FRONT_LIVINGROOM_FURNITURE)

print("bedroom_unique =", bedroom_unique)
print("library_unique =", library_unique)
print("livingroom_unique =", livingroom_unique)


with open(
    "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront/dataset_files/all_rooms_info.json",
    "w",
) as f:
    json.dump(
        {
            "bedroom": {
                "unique_values": bedroom_unique,
                "num_classes": len(bedroom_unique),
                "room_type": "bedroom",
                "max_objects": 12,
            },
            "library": {
                "unique_values": library_unique,
                "num_classes": len(library_unique),
                "room_type": "library",
                "max_objects": 21,
            },
            "livingroom": {
                "unique_values": livingroom_unique,
                "num_classes": len(livingroom_unique),
                "room_type": "livingroom",
                "max_objects": 21,
            },
        },
        f,
        indent=2,
    )
