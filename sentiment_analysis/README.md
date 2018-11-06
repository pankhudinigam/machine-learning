READ ME:

Start Reading at the start of main marked by '#main:'
Library Support Required: nltk, re, pickle, enchant
Input: Reviews for tagger training, List of Stopwords, List of Features for the given product or service
Output: cleaned_review.txt, Result Counters.txt

Abbreviations regarding comments:
SAB=> Same As Before
EWNS=>exactly what the name suggests
RMFMI: refer module for more info

Code Description:
Open the file containing reviews for training the tagger.
Open the list of Stop words.
Open the list of provided features.
Create a closure of the feature list.
Create a counter list which stores tuples as: [feature_name,pos_count,neg_count,neu_count] where pos_count represents the number of times the feature "feature_name" has been mentioned in the reviews in a positive sense ans so on.
read reviews
Handle content within double quotes in reviews
convert reviews to lower case
add space b4 each full stop in reviews
convert all occurences of n't to not
add '\n' at '.'
First Review empty=>So remove it
Read stop words
create a list of stopwords
Read features
Remove repetitive chars
Make not appear
Correct spelling mistakes
Implement the idea of Connected Features.
Implement the idea of Possessiveness.
Do Part of Speech Tagging
Write tagged reviews as output to file.
Writing counters to File.

Module Description:
prepare_feature_list()
prepare feature list using: synsets, synonyms,hypernyms,hyponyms etc.

double_quotes_handling(text)
if statement b/w double quotes too big then the first quote ignored, so as to make sure we dont lose important content by marking it as the text within double quotes(which generally represent conversations=>irrelevant for us, and in that case we ignore this content).

to_lower_case(text)
Converts input text to lower case.

add_space_b4_fullstop(text)
add a fullstop before space

pre_make_not_appear(text)
convert all instances of "n't" to "not"

add_newlines_at_full_stop(text)
EWNS;RMFMI

remove_repetitive_chars()
changes dazzzzzzzling to dazling.
Removes repetitive chars from all words in reviews.
creates an obj of class RepeatReplacer()=>this return a replacer
calls replace(word) function of the above object=>replace using this replacer

RepeatReplacer()
Represents Class definition of RepeatReplacer: refer to a python book for class reference.

replace(word)
Replaces repetitive characters with a character..refer to use of Regular Expressions in python.

make_not_appear()
SAB

correct_spelling(all_features)
changes dazling to dazzling.
Corrects Spelling errors in features

connected_feature(reviews)
Features of type "battery life" replaced by "battery%life" so that battery and life are not treated as separate features by any means.

possessiveness(all_features)
all phrases of the type: phone's battery, battery of phone, battery in phone, headset with phone denote possessiveness and are replaced by battery.

mapping()
map adjectives to nouns using basic "less distance" approach.

pos_tagging(all_features,features)
Intensive Backoff Tagging with Unigram,Bigram and Trigram Tagger...Accuracy=>99.23%
tag the words in input using this tagger plan
preparing to work on the polarity.
Now Identify features
Identify nouns
Map adjectives to nouns
Decide polarity for each mapping
Prepare to write the conclusions to output file

write_to_file(reviews)
Write reviews with list of adjectives, nouns and mapping for each sentence.

result_counters_to_file()
for each feature_name:
Write pos_count,neg_count and neu_count to file