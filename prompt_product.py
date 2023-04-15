import random
import os
import pickle


def read_words_from_file(file_path):
    with open(file_path, 'r') as f:
        words = [line.strip() for line in f]
    return words


def read_modifiers(base_dir='openprompts/modifiers'):
    modifiers_dict = {}
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            for subcategory_file in os.listdir(category_path):
                subcategory_file_path = os.path.join(category_path, subcategory_file)
                if os.path.isfile(subcategory_file_path) and subcategory_file.endswith('.txt'):
                    file_key = os.path.splitext(subcategory_file)[0]
                    modifiers_dict[file_key] = read_words_from_file(subcategory_file_path)
    return modifiers_dict


modifiers_dict = read_modifiers()

adjectives = [
    "tranquil", "enchanted", "moonlit", "vibrant", "majestic",
    "whimsical", "bustling", "serene", "idyllic", "mystical",
    "otherworldly", "ancient", "futuristic", "lush", "colorful",
    "dazzling", "dreamy", "ethereal", "forgotten", "haunted",
    "hidden", "isolated", "magical", "mysterious", "nautical",
    "peaceful", "quaint", "remote", "romantic", "ruined",
    "secluded", "surreal", "timeless", "untamed", "utopian",
    "verdant", "wild", "scenic", "radiant", "charming",
    "picturesque", "awe-inspiring", "grand", "spectacular", "breathtaking",
    "gorgeous", "stunning", "unusual", "unique", "intriguing",
    "extraordinary", "fascinating", "rugged", "barren", "ominous",
    "desolate", "forgiving", "harmonious", "graceful", "enchanted",
    "lively", "still", "gentle", "wondrous", "energetic",
    "dynamic", "vast", "endless", "infinite", "boundless",
    "limitless", "impressive", "sacred", "soaring", "towering",
    "precious", "abundant", "lavish", "rich", "ornate",
    "celestial", "delightful", "cozy", "warm", "welcoming",
    "elegant", "elusive", "exotic", "mesmerizing", "spellbinding",
    "captivating", "sublime", "nostalgic", "ancient", "evocative",
    "inspiring", "venerable", "mystifying", "arcane", "enigmatic",
]

subjects = [
    "forest", "beach", "mountain range", "waterfall", "desert",
    "garden", "cityscape", "castle", "village", "countryside",
    "tropical island", "coral reef", "river", "cave", "iceberg",
    "valley", "plateau", "marsh", "oasis", "volcano",
    "canyon", "swamp", "glacier", "lagoon", "prairie",
    "tundra", "cliff", "harbor", "ruins", "savanna",
    "island", "lake", "bay", "grotto", "peninsula",
    "rainforest", "archipelago", "badlands", "steppe", "dune",
    "fjord", "ravine", "gorge", "jungle", "underwater",
    "skyline", "farm", "vineyard", "orchard", "meadow",
    "rock formation", "wetland", "waterfront", "highlands", "desert oasis",
    "field", "delta", "reservoir", "atoll", "reef",
    "ancient city", "amusement park", "monument", "observatory", "port",
    "temple", "palace", "national park", "woodland", "aquarium",
    "botanical garden", "geyser", "factory", "library", "lighthouse",
    "old town", "bridge", "towers", "water wheel", "skyscraper",
    "market", "train station", "airport", "seaport", "statue",
    "fountain", "windmill", "zoo", "asylum", "colosseum",
]


environments = [
    "at sunrise", "at sunset", "under a starry sky", "during a thunderstorm", "with a rainbow",
    "shrouded in mist", "in the depths of winter", "in the peak of spring", "in the heat of summer",
    "amidst the colors of autumn", "bathed in moonlight", "under a blanket of snow",
    "in the calm after a storm", "illuminated by the Northern Lights", "under an alien sky",
    "in a dream world", "within a floating city", "on a distant planet", "in a magical realm", "on a mystical mountaintop",
    "in a parallel universe", "amidst a meteor shower", "during an eclipse", "under a solar flare", "beside a mirage",
    "on a cloud", "in a whirlwind", "within a sandstorm", "inside a crystal", "on the edge of reality",
    "beneath a waterfall", "inside a cave", "in a sunken city", "in an enchanted grove", "in a hidden valley",
    "on the ocean floor", "in a coral labyrinth", "at the edge of the world", "in the eye of a storm", "in a world of floating islands",
    "at the edge of space", "on a sky island", "in an underwater city", "within a dense forest", "in a fiery landscape",
    "in a frozen wasteland", "in the heart of a volcano", "amongst floating mountains", "in a city in the sky", "on a comet",
    "in a giant's garden", "in a mystical library", "on a pirate island", "in a sunlit glade", "in a crystal cavern",
    "within a giant tree", "on a moonlit beach", "in an endless maze", "in a futuristic cityscape", "at the bottom of a canyon",
    "on a mountaintop temple", "in a world made of glass", "in a world of eternal twilight", "in a place where time stands still", "on a lost continent",
    "in a realm of eternal night", "in a desert of shifting sands", "on a world of floating water", "in a city of eternal dawn", "on a planet of perpetual storms",
    "in an underwater cave", "in a place where dreams come true", "in a world of living shadows", "in a city that never sleeps", "on a world where the seasons never change",
    "in a place where gravity is reversed", "on a celestial battlefield", "in a city made of ice", "in a landscape of towering crystals", "on a world of endless flowers",
    "in a place where the stars are within reach", "in a world of eternal rain", "on a world where the sun never sets", "in a city built on clouds", "in a place where time flows backward",
    "on a world of perpetual twilight", "in a realm of floating islands", "in a place where the wind never stops", "in a city that moves with the wind", "in a world of infinite possibilities",
]


features = [
    "with a crystal-clear lake", "filled with exotic flowers", "populated by diverse wildlife",
    "interspersed with ancient ruins", "featuring towering rock formations", "dotted with charming cottages",
    "connected by winding pathways", "with a network of hidden tunnels", "surrounded by a misty aura",
    "reflecting on tranquil waters", "dominated by a grand palace", "home to a secret civilization",
    "guarded by mythical creatures", "adorned with intricate carvings", "encircled by a magical barrier",
    "with cascading waterfalls", "enveloped in a dense fog", "carpeted with lush vegetation", "overgrown with ivy and vines", "beside a serene pond",
    "covered in a blanket of moss", "inhabited by mystical beings", "with a skyline of towering spires", "marked by ancient standing stones", "protected by a ring of fire",
    "with a sky full of floating lanterns", "bordered by a bottomless chasm", "home to rare and magical flora", "traversed by a web of rope bridges", "illuminated by bioluminescent plants",
    "underneath a canopy of giant trees", "with an otherworldly atmosphere", "sheltered by towering cliffs", "infused with a sense of wonder", "surrounded by a seemingly endless ocean",
    "separated by a winding river", "visited by fantastical creatures", "with an air of mystery and intrigue", "reached by a hidden staircase", "filled with a sense of history",
    "where time seems to stand still", "with evidence of ancient battles", "littered with forgotten relics", "with an ever-changing landscape", "with paths that lead to unknown destinations",
    "carved into the side of a mountain", "floating on a sea of clouds", "built on a foundation of massive stones", "with a labyrinth of underground caverns", "with walls that tell a thousand stories",
    "nestled in the heart of a dense forest", "with an unearthly glow", "with a secret garden hidden within", "with a view that stretches for miles", "with skies that change color with the wind",
    "where the earth meets the sky", "with an otherworldly sense of beauty", "where magic seems to flow through the air", "where every corner holds a new discovery", "where whispers of the past can still be heard",
    "with a landscape that defies gravity", "with a view of a distant world", "where nature and technology coexist", "where ancient legends come to life", "where the spirits of the past still roam",
    "with a bridge that leads to nowhere", "with a tree that reaches the heavens", "with a hidden door to another realm", "where the elements converge", "with a touch of the surreal",
    "with an ever-present sense of wonder", "where the line between reality and fantasy is blurred", "with a sense of serenity that permeates the air", "where the days are bathed in a golden light", "with the scent of magic on the breeze",
    "with an aura of timeless beauty", "where the stars seem close enough to touch", "with a sense of awe that fills the soul", "where every breath is filled with wonder", "with a landscape that captures the imagination",
    "with a quiet sense of majesty", "with a beauty that transcends the physical world", "with an atmosphere that evokes a sense of peace", "where the heart of the world beats",
    "with a landscape that dances with the wind", "where natural formations create stunning patterns", "with a river that sings a timeless song", "where the sun and moon share the sky", "with the whispers of ancient spirits in the air",
    "where the trees tell stories of the past", "with a sky that paints a different picture every day", "where the wind carries the scent of adventure", "with a beauty that defies explanation", "where the wild and untamed reign supreme",
    "with a symphony of nature's sounds", "where the very stones hold secrets", "with a sense of the divine in every step", "where the veil between worlds is thin", "with an ever-changing tapestry of colors",
    "with a sense of harmony that resonates through the land", "where the forces of nature are unleashed", "with a beauty that humbles the soul", "where the wonders of the world are gathered", "with a sense of enchantment that lingers long after",
    "with a feeling of interconnectedness with all living things", "where the landscape tells a story of its own", "with a sense of awe that knows no bounds", "where the spirit of the land is alive and well", "with a beauty that reaches out and touches the heart",
]



time_periods = [
    "ancient", "medieval", "renaissance", "baroque", "Victorian",
    "modern", "futuristic", "prehistoric", "industrial revolution", "colonial",
    "post-apocalyptic", "morning", "afternoon", "evening", "night",
    "dawn", "dusk", "golden hour", "blue hour", "stone age",
    "bronze age", "iron age", "classical", "dark ages", "feudal",
    "enlightenment", "romantic", "impressionist", "Roaring Twenties", "Great Depression",
    "World War I", "World War II", "Cold War", "Space Age", "digital age",
    "information age", "age of exploration", "age of sail", "early civilizations", "ancient Egypt",
    "ancient Greece", "ancient Rome", "Byzantine", "Ottoman", "Mughal",
    "Ming Dynasty", "Qing Dynasty", "Aztec", "Inca", "Mayan",
    "Viking Age", "Babylonian", "Sumerian", "Assyrian", "Persian",
    "Hellenistic", "Carolingian", "Merovingian", "Anglo-Saxon", "Norman",
    "Plantagenet", "Tudor", "Stuart", "Hanoverian", "Napoleonic",
    "Belle Époque", "Edwardian", "Art Deco", "Art Nouveau", "Gilded Age",
    "Jazz Age", "Beat Generation", "Lost Generation", "counterculture", "Age of Aquarius",
    "Silent Generation", "Baby Boomer", "Generation X", "Millennial", "Generation Z",
    "Taisho", "Meiji", "Heian", "Kamakura", "Edo",
    "Joseon", "Goryeo", "Unified Silla", "Three Kingdoms", "Proto-Three Kingdoms",
]


colors = [
    "red", "blue", "green", "yellow", "purple",
    "orange", "pink", "brown", "black", "white",
    "gray", "gold", "silver", "bronze", "ivory",
    "teal", "turquoise", "maroon", "navy", "magenta",
]

actions = [
    "running", "flying", "swimming", "dancing", "sleeping",
    "eating", "reading", "singing", "painting", "jumping",
    "climbing", "riding", "driving", "fishing", "exploring",
    "gazing", "wandering", "strolling", "hiking", "crawling",
    "meditating", "crafting", "building", "writing", "sketching",
    "resting", "jogging", "stretching", "posing", "lifting",
    "fighting", "sparring", "performing", "practicing", "observing",
    "pursuing", "traversing", "sailing", "rowing", "diving",
    "soaring", "leaping", "twirling", "flipping", "cycling",
    "rolling", "balancing", "striding", "racing", "whispering",
    "laughing", "crying", "embracing", "arguing", "smiling",
    "frowning", "pondering", "daydreaming", "shouting", "chatting",
    "studying", "listening", "planning", "plotting", "scheming",
    "negotiating", "competing", "cooking", "baking", "mixing",
    "brewing", "sculpting", "carving", "weaving", "knitting",
    "sewing", "tending", "gardening", "harvesting", "plowing",
    "cultivating", "nurturing", "caressing", "kissing", "waving",
    "saluting", "bowing", "kneeling", "praying", "preaching",
    "teaching", "learning", "discovering", "uncovering", "inventing",
    "experimenting", "analyzing", "calculating", "measuring", "recording",
]


def random_choice_from_list(lst):
    return random.choice(lst)


def generate_prompt_with_conditional_structures():
    adjective = random_choice_from_list(adjectives)
    subject = random_choice_from_list(subjects)
    environment = random_choice_from_list(environments)
    feature = random_choice_from_list(features)
    time_period = random_choice_from_list(time_periods)
    color = random_choice_from_list(colors)
    action = random_choice_from_list(actions)


    # More sophisticated prompt structure
    prompt_components = [
        f"{adjective.capitalize()} {subject}",
        environment,
        feature,
        time_period,
        color,
        action,
    ]

    # Randomly select a number of components to include in the prompt
    num_components = random.randint(3, len(prompt_components))
    selected_components = random.sample(prompt_components, num_components)

    # randomly select modifiers
    num_modifiers = random.randint(1, len(modifiers_dict))
    selected_modifiers_keys = random.sample(list(modifiers_dict.keys()), num_modifiers)
    selected_modifiers = [random_choice_from_list(modifiers_dict[m]) for m in selected_modifiers_keys]

    prompt = ', '.join(selected_components + selected_modifiers)

    return prompt


# Generate a large number of prompts
random_seed = 20
random.seed(random_seed)
num_prompts = 1000000
generated_prompts = [generate_prompt_with_conditional_structures() for _ in range(num_prompts)]

# Print some example prompts
for i, prompt in enumerate(generated_prompts[:10]):
    print(f"Prompt {i+1}: {prompt}")

pickle.dump(generated_prompts, open(f"/remote-home/share/bxy/SD_2M/hardcoded_prompts_V{random_seed}.pk", "wb"))