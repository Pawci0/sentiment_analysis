import os
from nltk.corpus import stopwords
from string import punctuation

stop_words = set(stopwords.words('english'))


def parse_words(words):
    new_words = []
    for word in words:
        if word not in stop_words:
            if word[-1] in punctuation:
                new_words.append(word[:-1] + ' ' + word[-1])
            else:
                new_words.append(word)

    return new_words


def parse_lines(lines):
    utterances = []
    utterance_sentiments = []
    final_sentiment = lines.pop(-1).rstrip()
    for line in lines[:-1]:
        words = line.lower().split(' ')
        utterance_sentiments.append(words.pop(-1).rstrip())
        utterances.append(' '.join(parse_words(words[2:])))

    text = ' __eou__ '.join(utterances)
    sentiments = ' '.join(utterance_sentiments)

    return text, sentiments, final_sentiment


conversations = []
conversation_sentiments = []
conversation_final_sentiments = []
for root, dirs, files in os.walk('../data/ScenarioSA-raw'):
    for name in files:
        with open(os.path.join(root, name), 'r') as f:
            file_lines = f.readlines()
            text, sentiments, final_sentiment = parse_lines(file_lines)

            conversations.append(text)
            conversation_sentiments.append(sentiments)
            conversation_final_sentiments.append(final_sentiment)

with open('../data/ScenarioSA/dialogues_text.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(conversations))

with open('../data/ScenarioSA/dialogues_sentiment.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(conversation_sentiments))

with open('../data/ScenarioSA/dialogues_final_sentiment.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(conversation_final_sentiments))
