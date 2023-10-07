import time

from google_trans_new import google_translator
from googletrans import constants
import csv
from alive_progress import alive_bar

translator = google_translator()
'''
wordLst = ["be", "become", "do", "go", "come", "enter", "exit", "wait", "stay", "leave", "see something",
           "say something", "hear something", "touch", "smell something", "taste", "eat", "drink", "have",
           "hold something", "bring", "get something", "give", "send", "put", "take", "push", "pull", "open something",
           "close something", "begin something", "finish something", "lose something", "search for", "find", "buy",
           "sell", "lie down", "sit", "stand", "walk", "run", "jump"]

wordLst2 = ["is", "does", "goes", "comes", "enters", "exits", "leaves", "sees", "says", "hears", "touches", "smells",
           "tastes", "eats", "drinks", "has", "holds", "brings", "gets", "gives", "sends", "puts", "takes", "pushes",
           "pulls", "opens", "closes", "begins", "finishes", "loses", "searches", "finds", "buys", "sells", "lays",
           "sits", "stands", "walks", "runs", "jump"]
'''
# wordLst = ["I", "you", "he", "she", "it", "we", "you", "they", "this", "that", "here", "there", "who", "what", "where", "when", "how", "not", "all", "many", "some", "few", "other", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "woman", "man", "man", "child", "wife", "husband", "mother", "father", "animal", "cow", "horse", "pig", "mouse", "fish", "bird", "cat", "dog", "tiger", "lion", "rabbit", "sheep", "lamb", "monkey", "chicken", "insect", "ant", "bee", "butterfly", "louse", "snake", "worm", "plant", "tree", "forest", "stick", "fruit", "seed", "leaf", "root", "bark", "flower", "grass", "rope", "skin", "meat", "blood", "bone", "fat", "oil", "egg", "milk", "cheese", "bean", "powder", "fluid", "rice", "wheat", "bread", "soup", "wine", "fruit", "juice", "cake", "sugar", "food", "knife", "spoon", "fork", "plate", "bowl", "cup", "bottle", "pot", "horn", "tail", "feather", "fur", "hair", "beard", "head", "face", "ear", "eye", "tear", "nose", "mouth", "lip", "tooth", "tongue", "finger", "fingernail", "foot", "toe", "leg", "knee", "hand", "arm", "shoulder", "belly", "guts", "neck", "back", "breast", "heart", "liver", "brain", "sweat", "disease", "pain", "medicine", "poison", "body", "life", "world", "nature", "sun", "moon", "star", "air", "water", "wave", "bubble", "rain", "river", "lake", "sea", "beach", "island", "salt", "stone", "sand", "dust", "earth", "cloud", "lightning", "thunder", "storm", "fog", "sky", "wind", "snow", "ice", "smoke", "fire", "ash", "road", "hole", "swamp", "land", "field", "mountain", "desert", "cliff", "coast", "plain", "hill", "valley", "night", "day", "year", "season", "spring", "summer", "autumn", "winter", "light", "shadow", "mirror", "sound", "voice", "material", "glass", "metal", "plastic", "wood", "clay", "gold", "silver", "copper", "diamond", "silk", "cotton", "linen", "wool", "cloth", "clothes", "power", "war", "peace", "weapon", "arrow", "bow", "spear", "sword", "axe", "hook", "whip", "hammer", "rake", "trap", "shield", "gun", "scales", "name", "home", "house", "room", "wall", "floor", "ceiling", "window", "curtain", "roof", "pool", "garden", "yard", "door", "table", "chair", "bed", "soap", "lock", "key", "needle", "thread", "bag", "box", "card", "ring", "tool", "gift", "language", "word", "paper", "book", "page", "pencil", "pen", "brush", "ink", "letter", "note", "paint", "picture", "photograph", "toy", "clock", "lamp", "fan", "game", "sport", "race", "team", "music", "art", "city", "town", "bridge", "station", "school", "student", "market", "shop", "hotel", "restaurant", "park", "hospital", "office", "farm", "camp", "theater", "library", "temple", "port", "bank", "country", "building", "court", "prison", "loop", "edge", "corner", "line", "point", "shape", "circle", "triangle", "square", "rectangle", "ball", "time", "place", "space", "gap", "ground", "at", "with", "and", "but", "if", "because", "so", "when", "now", "yet", "still", "before", "after", "already", "never", "often", "again"]
#wordLst = ["to accept", "to deny", "to borrow", "to lend", "to edit", "first", "previous", "spleen", "soysauce", "stamp", "postcard", "sheet", "basket", "web", "law", "church", "weight", "structure", "distance", "displacement", "vector", "speed", "velocity", "direction", "magnitude", "acceleration", "force", "collision", "gravity", "inertia", "friction", "impulse", "pressure", "momentum", "particle", "atom", "electricity", "magnet", "ray", "telescope", "lens", "universe", "nebula", "galaxy", "black hole", "planet", "satellite", "meteor", "comet", "crater", "solar system", "eclipse", "solstice", "twilight", "dawn", "dusk", "phase", "zodiac", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "popular", "responsible", "serious", "formal", "casual", "thrifty", "sane", "rational", "dizzy", "crazy", "sober", "drunk", "wax"]

#wordLst = ["ready", "likely", "private", "public", "foreign", "local", "perfect", "part", "whole", "every", "each", "or", "except for", "although", "from", "to", "from here to there", "during", "around", "among", "between", "toward", "sure", "maybe", "even", "only", "also", "most", "least", "district", "rule", "amount", "pair", "media", "message", "view", "effort", "problem", "model", "performance", "analysis", "cause and effect", "evidence", "chance", "period", "a period of time", "goal", "standard", "idea", "knowledge", "detail", "feature", "case", "money", "system", "society", "environment",  "education", "culture", "politics", "military", "army", "agriculture", "company", "industry", "story", "job", "justice", "moral", "memory", "past, present, future", "to discuss", "to debate", "to explain", "to provide", "to fund", "to control", "to practice", "to prepare", "to challenge", "to affect", "to design", "to plan"]
#wordLst = ["to chat", "to understand", "to predict", "to assume", "to guess", "to respect", "to contempt", "to suffer", "to endure", "more", "less", "lotus", "eagle", "beverage", "ceramic", "sponge", "foam", "chain", "type", "information", "stripe", "grid", "important", "curious"]
#wordLst = ["to hope", "to donate", "to steer", "to guide", "to prefer", "to miss", "to scold", "to visit", "to lie", "to argue", "to fix", "to scoop", "to disdain", "to mention", "to explore", "to colonize", "to roam", "patient", "brave", "timid", "transparent", "positive", "negative", "independent", "dependent", "naked", "even", "odd", "single", "double", "familiar", "strange", "random", "stable", "joints", "crust", "layer", "snot", "rust", "torch", "responsibility", "treasure", "balloon", "value", "experience", "example", "file", "a file", "trace", "pivot", "spindle", "shuttle", "fiber", "yarn", "warp and weft", "loom", "beam", "pillar", "pipe", "straw", "list", "element", "finance", "layer", "cigarette", "saw", "compass", "ruler", "cicada", "aphid", "moth", "millipede", "cricket", "firefly", "herb", "sea anemone", "iris", "pupil", "hazelnut", "chilli", "entertainment", "wardrobe", "chip", "nest", "level", "morality", "history", "whirlpool", "table", "calendar", "schedule", "as"]
#wordLst = ["to ignore", "preference", "bias", "bare", "gem", "regretful", "regret", "happiness", "luck", "fate", "coincidence", "probability", "theory", "experiment", "economy", "mathematics", "physics", "chemistry", "biology", "literature", "coral", "paradox", "conflict", "arbitrary", "certain", "approximate", "exact", "precise", "can", "must", "should", "just", "just", "recent", "originally", "finally", "immediately", "process", "progress", "end", "consistent", "gradually", "suddenly", "rarely", "sometimes", "usually", "always", "all the time", "each other", "almost", "a little", "somewhat", "extremely", "forever", "average", "maximum", "minimum", "except", "according to", "for"]
wordLst = ["cane", "eyeglasses", "jade", "tidy", "meaning", "wealth", "carapace"]
#wordLst = ["to " + w for w in wordLst]

wordStr = '\n'.join(wordLst)
print(wordStr)
langLst = ['fi', 'el', 'ru', 'de', 'es', 'it', 'fr', 'ga', 'en', 'zh-cn', 'vi', 'ja', 'ko', 'th', 'id',
           'tr', 'ar', 'he', 'sw']
'''
langLst2 = ['fi', 'el', 'ru', 'de', 'nci', 'es', 'it', 'fr', 'ga', 'en', 'vi', 'ko',
            'id', 'tr', 'sw']

langLst2 = ['bod']

'''
langLst2 = ['hu', 'fi', 'el', 'ru', 'de', 'es', 'it', 'fr', 'ga', 'cy', 'en', 'vi', 'ko',
            'id', 'tr', 'sw', 'hi']

langLst2 = ['th']
wordCollection = []


def trans(lang):  # a word's translations in different languages
    if lang == 'en':
        output = wordStr.replace('\n', '  ')
    else:
        output = translator.translate(wordStr, lang_src='en', lang_tgt=lang).lower()
    print("Done with translations of " + lang + "!")
    return output


def dictOut():
    with alive_bar(len(langLst), force_tty=True) as bar:
        for lang in langLst:  # a dictionary made of word -> translations
            wordCollection.append(trans(lang))
            bar()
    print("Done with dictionary!")

    with open('collection.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        langLn = []
        for lang in langLst:
            langLn += [constants.LANGUAGES[lang].capitalize()]
        writer.writerow(langLn)
        wordCollLst = []
        for str in wordCollection:
            wordCollLst.append(str.lower().rstrip().split("  "))
        print(wordCollLst)
        wordCollLstT = zip(*wordCollLst)
        #print(list(map(list, wordCollLstT)))
        for words in list(map(list, wordCollLstT)):
            writer.writerow(words)

    print("Done with output!")


# dictOut()
