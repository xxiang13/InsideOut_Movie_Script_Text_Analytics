# -*- coding: utf-8 -*-
"""
MSiA - Text Analytics - Homework 4

@author: Steven Lin, Xiang Li, Yu Ang

# References
http://www.crummy.com/software/BeautifulSoup/bs3/documentation.html
http://www.crummy.com/software/BeautifulSoup/bs4/doc/
http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
http://rufuspollock.org/2006/09/25/unicode-to-ascii-mappings-for-standard-characters-from-wordprocessed-documents/
http://nlp.stanford.edu/software/corenlp.shtml
https://github.com/brendano/stanford_corenlp_pywrapper
http://www.nltk.org/api/nltk.classify.html
http://stackoverflow.com/questions/21980086/how-to-use-python-nltks-probdisti-class
http://www.nltk.org/book/ch06.html
http://streamhacker.com/2010/05/17/text-classification-sentiment-analysis-precision-recall/
http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/

# References
# http://www.nltk.org/book/ch03.html
# http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
# http://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file-using-python
# http://stackoverflow.com/questions/2225564/get-a-filtered-list-of-files-in-a-directory
# http://matthewrocklin.com/blog/work/2014/05/01/Fast-Data-Structures/
# http://pandas.pydata.org/pandas-docs/stable/dsintro.html
# http://pandas.pydata.org/pandas-docs/version/0.15.2/indexing.html
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html
# http://stackoverflow.com/questions/25736861/python-pandas-finding-cosine-similarity-of-two-columns
# http://stackoverflow.com/questions/28883303/calculating-similarity-between-rows-of-pandas-dataframe
# http://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
# http://stackoverflow.com/questions/19477264/how-to-round-numpy-values-in-savetxt
# http://stackoverflow.com/questions/5469286/how-to-get-the-index-of-a-maximum-element-in-a-numpy-array-along-one-axis

# References
# https://pypi.python.org/pypi/lda
# https://pythonhosted.org/lda/api.html
# http://chrisstrelioff.ws/sandbox/2014/11/13/getting_started_with_latent_dirichlet_allocation_in_python.html

# References
# http://pandas.pydata.org/pandas-docs/stable/gotchas.html
# http://brandonrose.org/clustering
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
# http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
# http://scikit-learn.org/stable/modules/clustering.html
# http://www.rdatamining.com/docs/text-mining-with-r-of-twitter-data-analysis

"""

from bs4 import BeautifulSoup
import re
from urllib2 import urlopen #python 2
import time
from __future__ import division # division results in float if both integers
import os
import json
import pandas
import copy
import numpy as np
from stanford_corenlp_pywrapper import CoreNLP
from nltk.corpus import stopwords
import nltk
from scipy.spatial.distance import cosine
import math
from sklearn.metrics import pairwise_distances
import sys
import numpy as np
import lda
import lda.datasets
import time
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
import scipy

path = '/Users/Steven/Documents/Academics/3_Graduate School/2014-2015_NU/MSIA_490_Text_Mining/hw/hw4'
in_file_name = 'inside_out.html'
out_file_name = 'inside_out.csv'
os.chdir(path)
os.listdir('.') # see if file is in directory


##############################################################################
#%% Function replace utf-8 “ and ’ character to ascii equivalent " and '

def processString(unicodeS):
    """ Convert ’ to ' and remove “, strip leading and trailing whitespace
    
    Args:
        unicdeS
    
    Returns:
        str: in ascii
        
    """
    out = unicodeS.replace( u'\u2018', u"'")
    out = out.replace( u'\u2019', u"'")
    out = out.replace( u'\u201c', '')
    out = out.replace( u'\u201d', '')
    out = out.replace('"','')
    return out.encode('ascii').strip()
    

#%% Data processing by div tag ###############################################

#url = 'http://webcache.googleusercontent.com/search?q=cache:zmFBXLVh38EJ:gointothestory.blcklst.com/wp-content/uploads/2015/11/Inside-Out.pdf+&cd=1&hl=en&ct=clnk&gl=us'
#html = urlopen(url)
bsObj = BeautifulSoup(open(in_file_name))

all_tags = bsObj.findAll("div")
len(all_tags) # 4817

# tag1: header, tag2: header, tag3: tags objects combined since there is not a closing "div" tag
# either use tag3:
len(all_tags[2].findAll("div")) # 4814

# or tag4 onwwards
len(all_tags[3:]) # 4814

all_tags = all_tags[3:]

# also ignore cover tags (e.g. Authors, phone number, etc)
all_tags = all_tags[10:]

# divs to ignore: 
#   2 divs in cover page
#   2 divs in header: INSIDE OUT, and page number
#   1 div for Page Number

# page number div tags have tag <b> and also attribute style with left:0
print(all_tags[0].prettify())
len([tag for tag in all_tags if tag.b !=None]) # 129 pages (excluding cover)

# headers have atttribute style with left:108 or left:810
print(all_tags[65].prettify())
print(all_tags[66].prettify())
pattern_head = re.compile(r'left:108|left:810')
len([tag for tag in all_tags if pattern_head.search(tag["style"])!=None]) #256, or 128 each (since second page has no header info)

# remove these div tags
margin_ignore = [0, 108, 810]
pattern_ignore = re.compile(r"{}".format("|".join(["left:" + str(i) for i in margin_ignore])))

tags1 = [tag for tag in all_tags if pattern_ignore.search(tag["style"]) == None]
len(tags1) #4419


# Alternative: process by span tag (page) #####################################
# tag span: attribute style="font-size:19px" is the header of page
# 130 pages, but first 2 pages don't have header (128 headers)
# finds all the page headers
ps = bsObj.findAll("span",style=re.compile('font-size:19px'))
len(ps) # 128

# get pages (represented by span tag), by ignoring span tags headers
pattern = re.compile('font-size:19px')
pages = bsObj.findAll("span", style=lambda tag: pattern.search(tag)==None)

# ignore cover
pages = pages[1:]
len(pages) #129

# do some testing
pages[0].attrs
pages[0].name
pages[0].div
pages[0]["style"]
pages[0].text
pages[0].findAll("nobr")
pages[0].contents # direct children
# generators
for x in pages[0].children:
    print x
for x in pages[0].descendants:
    print x
for x in pages[0].children:
    print x.string
for x in pages[0].strings:
    print x
for x in pages[0].stripped_strings :
    print x
pages[0].parent
print(pages[0].prettify())

# count number of divs
tags = [tag for page in pages for tag in page.findAll("div")]
len(tags) #4419 # this number matches data processing by div in previous section

tags2 = copy.copy(tags)


#%% Find Structure #############################################################

# get left margin
margins = [tag["style"].split(";")[2] for tag in tags]

series = pandas.Series(margins)
c = series.value_counts()
print(c) 

# left:270    1874
# left:162    1302
# left:378    1063
# left:309     131
# left:500      14
# left:324       6
# left:857       6
# left:302       4
# left:633       3
# left:618       3
# left:281       2
# left:291       2
# left:639       2
# left:693       2
# left:222       1
# left:641       1
# left:296       1
# left:629       1
# left:608       1


def removeTag(margin, tags):
    """ Given a margin for left, prints sample tags matching and returns tags
    with these margins removed
    
    Args:
        margin (int): margin left
        tags: list of div tag objects
    
    Returns:
        (list): of div tag object
    
    """
    tagsX = [tag for tag in tags if re.search("left:{}".format(margin), tag["style"])]
    print("Number of tags matching: " + str(len(tagsX)))
    print("Sample matching tags: ")
    print(tagsX[0:min(len(tagsX), 10)])
    
    margin_ignore2 = [margin]
    pattern_ignore2 = re.compile(r"{}".format("|".join(["left:" + str(i) for i in margin_ignore2])))
    tags2 = [tag for tag in tags if pattern_ignore2.search(tag["style"]) == None]
    
    return tags2
    
# Left:270 --> quotes
tags2 = removeTag(270,tags2)
# Left:162 --> scene location (uppercase) and description
# Note: some of the tags 162 (description and location) are quotes when
# multiple people speak at the same time
tags2 = removeTag(162,tags2)
# Left:378 --> character name
tags2 = removeTag(378,tags2)
# Left:309 --> description of quotes (in parenthesis) and also inside description during quote
tags2 = removeTag(309,tags2)
# Left:500 --> quotes, this occurs when multiple characters say around the same time
tags2 = removeTag(500,tags2)
# Left:324 --> description during a quote
tags2 = removeTag(324,tags2)
# Left:857 --> asterisk on the right margin
tags2 = removeTag(857,tags2)
# Left:302 --> character, when multiple characters say around the same time
tags2 = removeTag(302,tags2)
# Left:633 --> character, when multiple characters say around the same time
tags2 = removeTag(633,tags2)
# Left:618 --> character, when multiple characters say around the same time
tags2 = removeTag(618,tags2)
# Left:281 --> character, when multiple characters say around the same time
tags2 = removeTag(281,tags2)
# Left:291 --> character, when multiple characters say around the same time
tags2 = removeTag(291,tags2)
# Left:639 --> character, when multiple characters say around the same time
tags2 = removeTag(639,tags2)
# Left:693 --> CUT TO:
tags2 = removeTag(693,tags2)
# Left:222 --> character, when multiple characters say around the same time
tags2 = removeTag(222,tags2)
# Left:641 --> dissolve to:
tags2 = removeTag(641,tags2)
# Left:296 --> character, when multiple characters say around the same time
tags2 = removeTag(296,tags2)
# Left:629 --> character, when multiple characters say around the same time
tags2 = removeTag(629,tags2)
# Left:608 --> character, when multiple characters say around the same time
tags2 = removeTag(608,tags2)
    
# quote: 270, 500 (when multiple speakers), 162 (when multiple speakers)
# scene location: 162 (UPPER CASE)
# scene description: 162 (not all upper case)
# description quote: 309, 324 (second level)
# character: 378, multiple spearkers: 222, 281, 291,296, 302, 608, 618, 629, 633, 639
    
character_margins =[378, 222, 281, 291,296, 302, 608, 618, 629, 633, 639]

# "position:absolute;top:1470;left:162"   
# check that style contains 3 elements and the last one is left 
for tag in tags:
    if len(tag["style"].split(";")) !=3:
        print(tag)
    elif tag["style"].split(";")[-1].split(":")[0] != "left":
        print(tag)
        
pattern_margin = re.compile(r'[0-9]+$')

def getMargin(tag, pattern_margin):
    """ Returns the left margin number of a tag
    
    Args:
        tag: tag with style attribute
        
    Returns
        int: margin number
    """
    return int(pattern_margin.findall(tag["style"])[0])
      

# ignore left:693 (CUT TO) and left:857 (asterisk) and left: 641: DISSOLVE TO
tags = [tag for tag in tags if getMargin(tag, pattern_margin) not in (693, 857, 641)]

#%% Look at attribue top margin ###############################################
# get the differces in the top margin (vertical spacing) for pair of tags transitions

pattern_margin = re.compile(r'[0-9]+$')

def getMarginTop(tag, pattern_margin):
    """ Returns the top margin number of a tag
    
    Args:
        tag: tag with style attribute
        
    Returns
        int: margin number
    """
    return int(pattern_margin.findall(tag["style"].split(";")[1])[0])
        

marginTop = [getMarginTop(tag, pattern_margin) for tag in tags ]
differences = [t-s for (s, t) in zip(marginTop, marginTop[1:])]
marginLeft = [getMargin(tag, pattern_margin) for tag in tags]
tagsTransition = [str(t) + "->" + str(s) for (s, t) in zip(marginLeft, marginLeft[1:])]

tagDiff = zip(tagsTransition, differences)
tagDiff[0:20]

dftagsDiff = pandas.DataFrame({"tags": tagsTransition,"diff": differences}, columns = ["tags", "diff"])
dftagsDiff.groupby(["tags", "diff"]).size() # some are negating, indicating out of place tags

#%% Create Dataset #############################################################

data_raw = []    
count = 0
location = ""
description = ""
character = ""
quote = ""
current = ""

i=0
while i < len(tags) and tags[i].text != "THE END.":
    margin = getMargin(tags[i],pattern_margin)

    # location
    if margin == 162 and tags[i].text.isupper():
        
        if tags[i].text == "ON THE CONSCIOUSNESS SCREEN:":
            location = tags[i-1].text
            if re.search(r"CONTINUOUS$", location):
                location = re.sub(r" - CONTINUOUS$", "", location)
                  
        elif re.search(r"CONTINUOUS$", tags[i].text):
            #location = tags[i].text.split("-")[0].strip()
            location = re.sub(r" - CONTINUOUS$", "", tags[i].text)
            
        elif tags[i].text == "BEEP!":
            location = "INT. HEADQUARTERS"
            
        else:
            location = tags[i].text
            
        i+=1

    
    # description
    elif margin == 162:
        
        description = ""
        while margin == 162 and not tags[i].text.isupper() and i<len(tags):
            description += " " + tags[i].text
            i+=1
            margin = getMargin(tags[i],pattern_margin)
        
    # character
    elif margin in character_margins:
    #elif margin == 378:
        character = tags[i].text
        if tags[i+1].text[0] == "/":
            i+=1
            character = character + " " + tags[i].text
        
        # quote
        i+=1
        margin = getMargin(tags[i],pattern_margin)
        quote = ""
        while (margin in [270, 309, 324, 500] or 
        (margin == 162 and getMarginTop(tags[i],pattern_margin)-getMarginTop(tags[i-1],pattern_margin) == 18)) and i<len(tags)  :
                   
            quote += " " + tags[i].text
            i+=1
            margin = getMargin(tags[i],pattern_margin)
            
            if tags[i].text == "(MORE)":
                i+=1
                margin = getMargin(tags[i],pattern_margin)
            
        
        # remove (CONT'D) (O.S) (V.O) from character name        
        clean_character = re.sub(r'\(.*','',character).strip()
        
        data_raw.append({"location": processString(location),
                        "description": processString(description),
                        "character": processString(clean_character),
                        "quote": processString(quote)})
        
    else:
        i+=1
        
x = pandas.DataFrame(data_raw, columns = ["location", "description", "character", "quote"])
# x.to_csv(out_file_name, sep=",", header=True, index=True, encoding = 'utf-8')
         
#%% Clean Dataset #############################################################
         
# check character names         
pandas.Series(x["character"]).value_counts()         
characters = np.sort(pandas.unique(x["character"]))
for i in characters :
    print(i)
    
# find all characters
# table = bsObj.findAll("div", style=re.compile('left:378'))

# Some of the tags are our of place (e.g. top:XXX, when have (CONT'D) in character
# as a result of new page). This will usually show blank in quotes.
   
# names separted by / --> split in different row with same quote, location, etc

data_clean = []

index = 0
for row in data_raw:
    if row["quote"] != "":
        
        character_names = [ch.strip() for ch in row["character"].split("/")]
        
        for ch in character_names:
            character = ch
            data_clean.append({ "time": index,
                                "location": row["location"],
                                "description": row["description"],
                                "character": ch,
                                "quote": row["quote"]})
            index+=1
        
                
df_data_clean = pandas.DataFrame(data_clean, columns = ["time","character", "quote", "location", "description"])            
df_data_clean.to_csv("data_clean.csv", sep=",", header=True, index=False, encoding = 'utf-8')
      
with open('data_clean.json', 'w') as fp:
    json.dump(data_clean, fp, indent=2, encoding = 'utf-8')
    
#%% Replace Character Name and Standardize Locations, Add scenes  #############
    
# chracter counts
df_data_clean["character"].value_counts()

# locations 
df_data_clean["location"].value_counts().sort_index()

# change locations
# Standardize locations if contain
location_standard = ['TRAIN STATION','TRAIN CAR', 'SUBCONSCIOUS STAIRS',
                     'SUBCONSCIOUS GATE','SUBCONSCIOUS CAVE', 'STAGE B',
                     'SHCOOL', 'SAN FRANCISCO STREET', 'SAN FRANCISCO HOUSE',
                     "MOM'S HEADQUARTERS", 'MINNESOTA LAKE','MINNESOTA HOUSE',
                     'MINNESOTA HOSPITAL', 'MIND WORLD', 'MEMORY DUMP',
                     'LONG-TERM MEMORY', 'IMAGINATIONLAND', 'HOSPITAL', 
                     'HOCKEY RINK', 'HEADQUARTERS', 'GOOFBALL ISLAND',
                     'DREAM PRODUCTIONS', 'DO NOT ENTER WHEN LIGHT FLASHING.',
                     "DAD'S HEADQUARTERS",'CLASSROOM','BUS','ABSTRACT THOUGHT BUILDING']

# dictionary based on San Francisco or Minnesota to handle cases like "KITCHEN"
location_standard_dic_sf = {k:v for (k,v) in zip(location_standard,location_standard)} 
location_standard_dic_mn = {k:v for (k,v) in zip(location_standard,location_standard)} 

house_locations = ("LIVING ROOM", "DINING ROOM","BEDROOM", "KITCHEN", "BATHROOM", "RILEY'S ROOM")

for house in house_locations:
    location_standard_dic_sf[house] = 'SAN FRANCISCO HOUSE'
    location_standard_dic_mn[house] = 'MINNESOTA HOUSE'     

def standardizeLocation(s, dic):    
    """ Standardize Locations
    
    Args:
        s: string to standardie
        dic: dictionary key= pattern, value = to replace with
    Returns:
        string: standardized string or not match original string

    """        
    for loc in dic:
        if re.search(r'{}'.format(loc),s):
            return dic[loc]
    return s
    

# e.g. JOY will say SADNESS a lot, which is a negative sentiment
# Also NER will not be able to regonize Sadness as a person

# Mapping
# Joy --> Joyce
# Sadness --> Sandra
# Anger -->Angelo
# Fear --> Felipe
# Disgust --> Diane


emotion_to_person = {"Joy": "Joyce", "Sadness": "Sandra", "Anger": "Angelo",
                     "Fear": "Felipe", "Disgust": "Diane"}
                     
                     
person_to_emotion = {v: k for k,v in emotion_to_person.items()}


def convertEmotionToPerson(s, emotion_to_person):
    """ Convert the emotion to a person name
    
    Args:
        s (string): string to convert
        emotion_to_person (dic): mapping emotion to name
    
    Returns:
        string: with replaced emotions with names
    
    """
    
    for (emotion,person) in emotion_to_person.items():

        s = re.sub(r"{upper}|{regular}".format(upper = emotion.upper(),
                                               regular = emotion),person, s)
                   
    return s
        
data_names = copy.deepcopy(data_clean) # deep copy otherwise change data_clean since list of objects
for row in data_names:
    
    # in minnesota
    if row["time"] < 65:    
        row["location"] =standardizeLocation(row["location"], location_standard_dic_mn)
    
    # in san francisco
    else:
        row["location"] =standardizeLocation(row["location"], location_standard_dic_sf)
        
    for field in ("character","quote","description"):
        row[field] = convertEmotionToPerson(row[field], emotion_to_person)

print(data_names[0])
print(data_clean[0])

df_data_names = pandas.DataFrame(data_names, columns = ["time","character", "quote", "location", "description"])            
df_data_names.to_csv("data_names.csv", sep=",", header=True, index=False, encoding = 'utf-8')
      
with open('data_names.json', 'w') as fp:
    json.dump(data_names, fp, indent=2, encoding = 'utf-8')
    
# chracter counts
df_data_names["character"].value_counts()

# location counts
df_data_names["location"].value_counts()

###### Add scenes

# scene: index range
# 0: 0-64 (Riley is Born)
# 1: 65-292 (Moving to San Francisco)
# 2: 293-385 (Riley's First Day)
# 3: 386-458 (Dinner Argument)
# 4: 459-537 (Try to escape from Long-Term Memory)
# 5: 538-698 (Meet Bing Bong)
# 6: 699-750 (Hockey Tryouts)
# 7: 751-898 (Dream Productions)
# 8: 899-982 (Fall in Memory Dump)
# 9: 983-1056 (Back to Headquarters)
# 10: 1057-1120 (Riley is Back)

data_scenes = copy.deepcopy(data_names)

start_index = [0, 65, 293, 386, 459, 538, 699, 751, 899, 983, 1057]
scene_name = ["Riley is Born","Moving to San Francisco","Riley's First Day",
              "Dinner Argument","Try to escape from Long-Term Memory",
              "Meet Bing Bong", "Hockey Tryouts", "Dream Productions",
              "Fall in Memory Dump", "Back to Headquarters",
              "Riley is Back"]
              
dic_startindex_scene = {k: v for (k,v) in zip(start_index, scene_name)}
dic_scene_name = {k: v for (k,v) in enumerate(scene_name)}

# how many times each index is repeated
repeats = np.array(start_index + [len(data_scenes)])[1:]-np.array(start_index + [len(data_scenes)])[:-1]

# index of scene based on start_index eg. 0,0,0...65 times, 1,1,1,....293 times
scene_number = []
i = 0
for j in repeats:
    scene_number += j*[i]
    i+=1

for row in data_scenes:
    row["scene_number"] = scene_number[row["time"]]
    row["scene_name"] = dic_scene_name[row["scene_number"]]


df_data_scenes= pandas.DataFrame(data_scenes, columns = ["time","scene_number", "scene_name", "character", "quote", "location", "description"])            
df_data_scenes.to_csv("data_scenes.csv", sep=",", header=True, index=False, encoding = 'utf-8')
      
with open('data_scenes.json', 'w') as fp:
    json.dump(data_scenes, fp, indent=2, encoding = 'utf-8')
    
    

#%% Lemmatize ################################################################

# Note: Stanford NLP does not have stop word removal. So have to remove in python code using
# stopwords from nltk

# Note: Stanford NLP converts to lower cases regulars words (e.g does not 
# convert lower case I, China, etc). So need to add "I" to stopwords

# Note: Stanford NLP: parentheis are -rrb- and -lrb-

# stopwords list , add "I" since Stanford NLP does not lowercase I but stopwords
# from nltk includes "i"
stop = set(stopwords.words('english'))
stop.add("I")

# regex: only keep words composed of alphanumeric characters or alphanumeric or ! or ?
# words joined by "-" (e.g. keep data-driven)
# ignore parenthesis -rrb-, -lrb- so use match instead of search
# match: Determine if the RE matches at the beginning of the string.
pattern = re.compile(r'^(?:[A-Za-z0-9]+[- ][A-za-z0-9]+|[A-Za-z0-9]+|[?!]+)$')
#pattern_parenthesis = re.compile("-rrb-|-lrb-")


proc = CoreNLP("pos", corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])

# You can also specify the annotators directly. For example, say we want to 
# parse but don't want lemmas. This can be done with the configdict option:
# no longer need to specify output_types (the outputs to include are inferred from the annotators setting
p = CoreNLP(configdict={'annotators':'tokenize, ssplit, pos, parse, lemma, ner,entitymentions, dcoref'}, 
            #output_types=['pos','parse'],
            corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])
            
            
data_lemmas = copy.deepcopy(data_scenes) # deep copy otherwise change data_clean since list of objects

# lemmatize quotes and description
for row in data_lemmas:
    # Now it's ready to parse documents. You give it a string and it returns JSON-safe data structures
    # dictionary key = 'sentences', value = list of sentences
    # each sentence dictionary with key='lemmas', 'tokens', etc
    # key = 'lemmas', value = list of lemmas 
    
    for field in ("quote","description"):
        parsed = proc.parse_doc(row[field])["sentences"]
        sentences = [sentence["lemmas"] for sentence in parsed]
        
        # flatten nested list so each element is a token
        row_tokenized = [token.strip() for sentence in sentences for token in sentence
                         if token.strip() not in stop and pattern.match(token.strip())]
        row_string = " ".join(row_tokenized)
        row[field] = row_string
        #row[field] = row_tokenized           
        
print(data_scenes[0])
print(data_lemmas[0])

df_data_lemmas = pandas.DataFrame(data_lemmas, columns = ["time","scene_number", "scene_name","character", "quote", "location", "description"])            
df_data_lemmas.to_csv("data_lemmas.csv", sep=",", header=True, index=False, encoding = 'utf-8')
      
with open('data_lemmas.json', 'w') as fp:
    json.dump(data_lemmas, fp, indent=2, encoding = 'utf-8')

##############################################################################
#%% Bag of words ##############################################################

# combine quotes-description (because want to remove duplicates
# that were generated when people speak at the same time)

combine =  df_data_lemmas["quote"] + "@" + df_data_lemmas["description"]
len(combine)
combine_unique = list(set(combine)) #1073

# alternative
# remove duplicates columns quote and description, keep first (default: Drop duplicates except for the first occurrence)
df_data_lemmas_unique = copy.deepcopy(df_data_lemmas)
df_data_lemmas_unique.drop_duplicates(['quote','description'], inplace=True) # Whether to drop duplicates in place or to return a copy


quotes_token = df_data_lemmas_unique ["quote"].map(lambda x: x.split(" "))
quotes_token_flat = [token for q in quotes_token for token in q]

descriptions_nodup = df_data_lemmas_unique.drop_duplicates(['description'])["description"]
descriptions_token = descriptions_nodup .map(lambda x: x.split(" "))
descriptions_token_flat = [token for d in descriptions_token for token in d]

quotes_descriptions_token_flat = quotes_token_flat + descriptions_token_flat

# these are the inputs for the word cloud
with open('quotes_token.txt', 'w') as fp:
    for q in quotes_token_flat:
        fp.write(q+'\n')
        
with open('descriptions_token.txt', 'w') as fp:
    for d in descriptions_token_flat:
        fp.write(d+'\n')
        
with open('quotes_descriptions_quotes_token.txt', 'w') as fp:
    for d in quotes_descriptions_token_flat:
        fp.write(d+'\n')      

#%% Create base data #########################################################

# based on quotes or quotes+descriptions
# document = character
# document = index (aggregate)
# document = location

# remove punctuation except !
# keep uppercase


#%% Testing

fdist1 = nltk.FreqDist(" ".join(descriptions_token_flat))
fdist1.most_common(50)
fdist1["Joyce"]



# Reference for pandas dataframe
# iloc --> by row number/col number, loc --> by index 
# df_data_clean.iloc[0]
# df_data_clean.iloc[0:10]
# df_data_clean.iloc[0:10, 2] # rows 0 to 10, column 2
# df_data_clean["quote"].loc[0] # index 0 for column "quote"
# df_data_clean["quote"][0:10] # rows 0 to 10 for column "quote"

quotes = df_data_clean["quote"].loc[0] 

##############################################################################
#%% Document Similarity ######################################################

#%% Functions ###############################################################
    
def getCounts(doc):
    """ Gets the frequencies for elements in document
    
    args:
        doc (list) = list of tokens in document
    returns:
        pandas.core.series.Series: index = element, value = count
    """
    series = pandas.Series(doc)
    c = series.value_counts()
    return c
    
def getTermDocument(doc_dic_normalized):
    """ Term Document Dataframe, with column = document name, row = terms
    
    args:
        doc_dic_normalized (dic) = key: document name, value: list of terms
    returns:
        pandas.DataFrame with index = terms, column name = document name
    """    
       
    # key = document_name, value = pandas series counts
    counts = {name: getCounts(doc) for (name,doc) in doc_dic_normalized.items()}
    
    # Vocabulary = distinct words from all documents
    vocabulary = set([v for k in counts for v in counts[k].index])
    len(vocabulary) # 1769
    
    # for each class, fill missing values for words of that class that are not in the vocabulary
    # do this by creating a pandas series from the current count series using vocabulary
    # as index. The resulting series will include all the words in the vocabulary as index
    # with value equal to the count in the orignal series, except for words that are not
    # in the original series which will get a NaN value, which in turns gets filled with zero
    counts_fill = {k: pandas.Series(v, index = vocabulary).fillna(0, inplace=False) for k,v in counts.items()}
    
    # actually previous step not needed, can include vectors with different elements
    # in data frame, and any missing elements for other documents will have NaN values (fill with zeros)
    
    # column name = document name, row names = token
    # df = pandas.DataFrame(counts , index = vocabular).fillna(0, inplace = False) # no need to provide index
    df = pandas.DataFrame(counts).fillna(0, inplace = False)
    
    return df


def convertBoolean(term_document):
    """ Term Document Binarized (counts > 1 have value 1, else 0)
    
    args:
        term_document (pandas.DataFrame): with index = terms, column name = document name
    returns:
        pandas.DataFrame with index = terms, column name = document name
    """    
    
    term_document_boolean = term_document.copy(deep = True)
    term_document_boolean[term_document_boolean>1] = 1
    
    return term_document_boolean
 
def findMaxPairs(df):
    """ Returns the most similar document for each document
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
    
    returns:
        pandas.Dataframe: most similar document for each document
    
    """
    df2 = df.copy(deep=True)
    np.fill_diagonal(df2.values, -1)
    return df2.idxmax()
    
def findMinPairs(df):
    
    """ Returns the least similar document for each document
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
    
    returns:
        pandas.Dataframe: least similar document for each document
    
    """
    return df.idxmin() 
    
 
def rankSimilarity(df, top = True, rank = 3):
    
    """ Returns the most similar documents or least similar documents
    
    args:
        df (pandas.Dataframe): row, col = documents, value = boolean similarity
        top (boolean): True: most, False: least (default = True)
        rank (int): number of top or bottom (default = 3)
    
    returns:
        pandas.Dataframe: row =rank, columns = indices, names, value
    
    """
    df2 = df.copy(deep = True)
    df_np = df2.as_matrix()
    
    if top:
        np.fill_diagonal(df_np, -1)
    
    results_dic = {"indices": [], "names": [], "value": [] }
    
    for n in range(rank):
        
        if top:
            indices = np.unravel_index(df_np.argmax(), df_np.shape) # returns indices of first max found
            # np.where(df_np == df_np.max()) # will return all indices of maxs
        else:
            indices = np.unravel_index(df_np.argmin(), df_np.shape) # returns indices of first min found
            # np.where(df_np == df_np.min()) # will return all indices of mins
            
        results_dic["indices"].append(indices)
        results_dic["names"].append((df.index[indices[0]], df.index[indices[1]]))
        results_dic["value"].append(df.iloc[indices])
        
        if top:
            df_np[indices[0],indices[1]] = -1 # set to -1 to find the next max
            df_np[indices[1],indices[0]] = -1 # because symmetric
        
        else:
            df_np[indices[0],indices[1]] = 1 # set to 1 to find the next min
            df_np[indices[1],indices[0]] = 1 # because symmetric
        
    df_result = pandas.DataFrame(results_dic, index = range(1,rank+1))  
    df_result.index.name = "rank"   
    
    return df_result
    
def booleanSimilarity(doc_dic_normalized, k = 10):
    """ Boolean Similarity Analysis
    Args:
        doc_dic_normalized (dict): key= document name, value = list of lemmas
        top (int): top k and bottom similar documents
    Returns:
        dict:{"term_document": term_document,
              "term_document_boolean":term_document_boolean,
              "df_similarity_boolean": df_similarity_boolean,
              "max_pairs": max_pairs,
              "min_pairs": min_pairs,
              "topSimilar": topSimilar,
              "bottomSimilar": bottomSimilar}
    """
    
    # document-term matrix 
    term_document = getTermDocument(doc_dic_normalized)
    term_document.index 
    term_document.columns
    term_document.shape # (1289, 52)
    
    term_document_boolean = convertBoolean(term_document) # term's raw frequency in this document

    # Note that spatial.distance.cosine computes the distance, and not the similarity.
    # So, you must subtract the value from 1 to get the similarity.
    # Note also need to transpose because pairwise distances does it row by row
    term_document_boolean_tp = term_document_boolean.transpose()
    similarity_boolean = 1-pairwise_distances(term_document_boolean_tp, metric="cosine")
    np.shape(similarity_boolean) # 30 by 30
    
    df_similarity_boolean = pandas.DataFrame(similarity_boolean , index=term_document_boolean.columns, columns=term_document_boolean.columns)
    document_name_check = df_similarity_boolean.index[0]
    document_name_check2 = df_similarity_boolean.index[-1]
    df_similarity_boolean[document_name_check][document_name_check] # 1
    df_similarity_boolean[document_name_check][document_name_check2] # 0.1856
    
    max_pairs = findMaxPairs(df_similarity_boolean)
    min_pairs = findMinPairs(df_similarity_boolean)
    max_pairs
    min_pairs
    
    max_pairs[document_name_check] # 'bhargava_anuj'
    min_pairs[document_name_check] # 'leboeuf_mark'
    
    #map_name_index[document_name_check] # 15
    #map_name_index[max_pairs[document_name_check]] # 2
    #map_name_index[min_pairs[document_name_check]] # 13
    
    topSimilar = rankSimilarity(df_similarity_boolean, top = True, rank =k) 
    bottomSimilar = rankSimilarity(df_similarity_boolean, top = False, rank =k)
    
    return {"term_document": term_document,
            "term_document_boolean":term_document_boolean,
            "df_similarity_boolean": df_similarity_boolean,
            "max_pairs": max_pairs,
            "min_pairs": min_pairs,
            "topSimilar": topSimilar,
            "bottomSimilar": bottomSimilar}

def tfidfSimilarity(doc_dic_normalized, k = 10):
    """ TFIDF Similarity Analysis
    Args:
        doc_dic_normalized (dict): key= document name, value = list of lemmas
        top (int): top k and bottom similar documents
    Returns:
        dict:{"term_document": term_document,
              "term_document_boolean":term_document_boolean,
              "df_similarity_tfidf": df_similarity_tfidf
              "max_pairs": max_pairs,
              "min_pairs": min_pairs,
              "topSimilar": topSimilar,
              "bottomSimilar": bottomSimilar
              "tf_idf": tf_idf}
    """
    
    # document-term matrix 
    term_document = getTermDocument(doc_dic_normalized)
    term_document.index 
    term_document.columns
    term_document.shape # (1289, 52)
    
    term_document_boolean = convertBoolean(term_document) # term's raw frequency in this document

    # term's raw frequency in this document
    tf = 1+np.log10(term_document) # 1+log(tf) if tf > 0
    tf = tf.replace([-np.inf],0)   # 0 otherwise
    
    # the number of documents the term occurs in
    # index = term, value = count documents
    df = term_document_boolean.sum(axis=1)
    
    # check
    term_name_check = df.index[0]
    assert sum(term_document_boolean.loc[term_name_check,:]) == df[term_name_check]
    
    # N the total number of documents
    N = term_document.shape[1]
    
    idf = np.log10(N/df)
    tf_idf = tf.multiply(idf,axis = 'index')
    
    # check
    document_name_check = tf_idf.columns[0]
    assert sum(tf_idf[document_name_check]) == sum(tf[document_name_check]*idf)
    assert tf_idf.loc[term_name_check,document_name_check ] == tf.loc[tf_idf.index[0],document_name_check ]*idf[term_name_check]
    
    # Note that spatial.distance.cosine computes the distance, and not the similarity.
    # So, you must subtract the value from 1 to get the similarity.
    # Note also need to transpose because pairwise distances does it row by row
    term_document_tfidf_tp = tf_idf.transpose()
    similarity_tfidf = 1-pairwise_distances(term_document_tfidf_tp, metric="cosine")
    np.shape(similarity_tfidf) # 30 by 30
    
    document_name_check2 = tf_idf.columns[-1]
    df_similarity_tfidf = pandas.DataFrame(similarity_tfidf , index=tf_idf.columns, columns=tf_idf.columns)
    df_similarity_tfidf[document_name_check][document_name_check] # 1
    df_similarity_tfidf[document_name_check][document_name_check2] # 0.048035582616980377
    
    # save results
    #np.savetxt("tf_idf.txt", similarity_tfidf , fmt='%.3f', delimiter = " ")
    #df_similarity_tfidf.to_csv("tf_idf_w_headers.csv", sep=",", header=True, index=True, float_format= '%.3f')
                
    max_pairs = findMaxPairs(df_similarity_tfidf)
    min_pairs = findMinPairs(df_similarity_tfidf)
    max_pairs
    min_pairs
    
    max_pairs[document_name_check] # 'bhargava_anuj'
    min_pairs[document_name_check] # 'fassois_demetrios'
    
    #map_name_index[document_name_check] # 15
    #map_name_index[max_pairs[document_name_check]] # 2
    #map_name_index[min_pairs[document_name_check]] # 6
            
    topSimilar = rankSimilarity(df_similarity_tfidf, top = True, rank =k) 
    bottomSimilar = rankSimilarity(df_similarity_tfidf, top = False, rank =k)
    
    return {"term_document": term_document,
            "term_document_boolean":term_document_boolean,
            "df_similarity_tfidf": df_similarity_tfidf,
            "max_pairs": max_pairs,
            "min_pairs": min_pairs,
            "topSimilar": topSimilar,
            "bottomSimilar": bottomSimilar,
            "tf_idf": tf_idf}

def stackDF(df, column_names = ["column1","column2","values"], symmetric = True  ):
    """Stacks similarity dataframe by return column1, column2, values df
    
    Args:
        df: pandas data frame
        column_names: names of return df (default ["column1","column2","values"] )
        symmetric (boolean): true if symmetric and remove duplicates (default = True)
        
    Returns:
        pandas data frame: column1, column2, values df
    """
    
    n_rows = df.shape[0]
    n_cols = df.shape[1]
    
    column1 = []
    column2 = []
    values = []
    
    for column_name in df.columns:
        column1 += n_rows*[column_name]
        column2 +=  list(df.index)
        values += list(df[column_name])
        
    assert len(values) == n_rows*n_cols
    
    column_values = [column1, column2, values]
    dic_stacked = {k: v for (k,v) in zip(column_names, column_values)}
    df_stacked = pandas.DataFrame(dic_stacked, columns = column_names)

    if symmetric:
        
        # remove duplicates if symmetirc
        # create duplicate column (combination of column 1 and column 2)
        # first concatenate, then create to list, and sort, convert to string
        df_stacked["dup_column"] = df_stacked[column_names[0]] + "@" + df_stacked[column_names[1]]
        df_stacked["dup_column"] = df_stacked["dup_column"].map(lambda x: str(sorted(x.split("@"))))
        
        df_stacked.drop_duplicates(['dup_column'], inplace=True)
        # drop column
        df_stacked.drop('dup_column', axis=1, inplace=True)
    
        
        # check if remove one half of symmetric matrix is ok 
        assert (len(values) - len(df.columns))/2 + len(df.columns) == df_stacked.shape[0]
    
    return df_stacked

            
#%% Manual calculations, less efficient #######################################
# check cosine formula
"""
x = term_document_boolean["lin_luis"]
y = term_document_boolean["li_xiang"]
sum(pandas.Series.multiply(x,y)) #20
np.dot(x, y) #20
sum(i*j for (i,j) in zip(x,y)) #20
math.sqrt(sum(i**2 for i in x)) # 12.6
math.sqrt(sum(i**2 for i in y)) # 8.54
sum(i*j for (i,j) in zip(x,y))/(math.sqrt(sum(i**2 for i in x))*math.sqrt(sum(i**2 for i in y)))
np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
1 - cosine(x,y)
"""

# do manually
"""
m = term_document_boolean.shape[1]
mat = np.zeros((m, m))
np_term_document_boolean = term_document_boolean.as_matrix()
np.shape(np_term_document_boolean)
for i in range(m):
    for j in range(i,m):
        if i != j:
            mat[i][j] = 1 - cosine(np_term_document_boolean[:,i], np_term_document_boolean[:,j])
            mat[j][i] = mat[i][j]
        else:
            mat[i][j] = 1
            
np.sum(mat-similarity_boolean) # basically zero
"""   
        
#%% Analysis by document name = character, documents = quotes #################

fileSuffix = "quotes_by_char"

freq_char = df_data_lemmas.groupby('character')['quote'].count().order(ascending = False)
print(freq_char)

# save results
freq_char.to_csv("freq_char.csv", sep=",", header=True, index=True)

# this converts to dataframe with column1 = character, column2 = list of quotes
df_data_lemmas.groupby('character')['quote'].apply(lambda x: x.tolist()).iloc[0]

# this convert dataframe with column1 = character, column2 = list of tokens 
# (concatenates quotes with space in between then split to token)
# need to remove the last element of list because extra space as a result of concatenation
quotes_by_char = df_data_lemmas.groupby('character')['quote'].apply(lambda x: (x + " ").sum().split(" ")[:-1])
quotes_by_char = quotes_by_char.map(lambda x: [token for token in x if token != '']) # remove "" tokens
type(quotes_by_char) # series

# key: document name (character), value = list of lemmas
doc_dic_normalized  = quotes_by_char.to_dict()

# key: name, value = index number
map_name_index = {name: index for (name,index) in zip(quotes_by_char.index, range(len(quotes_by_char.index)))}


#%% Boolean Similarity ##########################################################

boolean_similarity = booleanSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("boolean_"+fileSuffix+".txt", boolean_similarity["df_similarity_boolean"] , fmt='%.3f', delimiter = " ")
boolean_similarity["df_similarity_boolean"].to_csv("boolean_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
boolean_similarity["topSimilar"].to_csv("boolean_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
boolean_similarity["bottomSimilar"].to_csv("boolean_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])
print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on characters with at least 20 quotes

freq_char[freq_char>20]
top_char = freq_char[freq_char>20].index

df_similarity_boolean_top = boolean_similarity["df_similarity_boolean"].loc[top_char,top_char]

topSimilar = rankSimilarity(df_similarity_boolean_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_boolean_top, top = False, rank =10)

# save results
topSimilar.to_csv("boolean_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("boolean_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))


boolean_similarity_quotes_by_char = copy.deepcopy(boolean_similarity)


stacked_boolean_similarity_quotes_by_char = stackDF(boolean_similarity_quotes_by_char["df_similarity_boolean"],
                                              ["character1", "character2", "boolean_similarity"], symmetric = True)


stacked_boolean_similarity_quotes_by_char.to_csv("boolean_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)



#%% tf_idf ##################################################################

   
tfidf_similarity = tfidfSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("tfidf_"+fileSuffix+".txt",tfidf_similarity["df_similarity_tfidf"] , fmt='%.3f', delimiter = " ")
tfidf_similarity["df_similarity_tfidf"].to_csv("tfidf_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
tfidf_similarity["topSimilar"].to_csv("tfidf_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
tfidf_similarity["bottomSimilar"].to_csv("tfidf_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on characters with at least 20 quotes

freq_char[freq_char>20]
top_char = freq_char[freq_char>20].index

df_similarity_tfidf_top = tfidf_similarity["df_similarity_tfidf"].loc[top_char,top_char]

topSimilar = rankSimilarity(df_similarity_tfidf_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_tfidf_top, top = False, rank =10)

# save results
topSimilar.to_csv("tf_idf_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("tf_idf_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

tfidf_similarity_quotes_by_char = copy.deepcopy(tfidf_similarity)

stacked_tfidf_similarity_quotes_by_char = stackDF(tfidf_similarity_quotes_by_char["df_similarity_tfidf"],
                                              ["character1", "character2", "tfidf_similarity"], symmetric = True)


stacked_tfidf_similarity_quotes_by_char.to_csv("tfidf_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)


#%% Analysis by document name = location, documents = quotes+description #####

fileSuffix = "quotes_loc_by_loc"

# combine quotes-description (because want to remove duplicates
# that were generated when people speak at the same time)
combine =  df_data_lemmas["quote"] + "@" + df_data_lemmas["description"]
len(combine)
combine_unique = list(set(combine)) #1073

# alternative
# remove duplicates columns quote and description, keep first (default: Drop duplicates except for the first occurrence)
df_data_lemmas_unique = copy.deepcopy(df_data_lemmas)
df_data_lemmas_unique.drop_duplicates(['quote','description'], inplace=True) # Whether to drop duplicates in place or to return a copy


freq_loc = df_data_lemmas_unique.groupby('location')['location'].count().order(ascending = False)
print(freq_loc )

# save results
freq_loc.to_csv("freq_loc.csv", sep=",", header=True, index=True)

# this converts to dataframe with column1 = character, column2 = list of quotes
df_data_lemmas_unique.groupby('location')['quote'].apply(lambda x: x.tolist()).iloc[0]

# this convert dataframe with column1 = character, column2 = list of tokens 
# (concatenates quotes with space in between then split to token)
# need to remove the last element of list because extra space as a result of concatenation
quotes_by_location = df_data_lemmas_unique.groupby('location')['quote'].apply(lambda x: (x + " ").sum().split(" ")[:-1])
quotes_by_location = quotes_by_location.map(lambda x: [token for token in x if token != '']) # remove "" tokens
type(quotes_by_location) # series

descriptions_nodup = df_data_lemmas_unique.drop_duplicates(['description'])

# this converts to dataframe with column1 = character, column2 = list of quotes
descriptions_nodup.groupby('location')['description'].apply(lambda x: x.tolist()).iloc[0]

# this convert dataframe with column1 = character, column2 = list of tokens 
# (concatenates quotes with space in between then split to token)
# need to remove the last element of list because extra space as a result of concatenation
descriptions_by_location = descriptions_nodup.groupby('location')['description'].apply(lambda x: (x + " ").sum().split(" ")[:-1])
descriptions_by_location = descriptions_by_location.map(lambda x: [token for token in x if token != '']) # remove "" tokens
type(descriptions_by_location) # series

# join series by location, column 1 = quotes, column2 = descriptions
quotes_descriptions_by_location = pandas.concat([quotes_by_location, descriptions_by_location], axis=1, join='inner')
quotes_descriptions_by_location.shape

# final data is a dataframe with index the location, column = quote_description as list of lemmas
quotes_descriptions_by_location["quote_description"] = quotes_descriptions_by_location["quote"] + quotes_descriptions_by_location["description"]
quotes_descriptions_by_location.drop('quote', axis=1, inplace=True)
quotes_descriptions_by_location.drop('description', axis=1, inplace=True)
# need series for dictionary
type(quotes_descriptions_by_location["quote_description"])


# key: document name (character), value = list of lemmas
doc_dic_normalized  = quotes_descriptions_by_location["quote_description"].to_dict()

# key: name, value = index number
map_name_index = {name: index for (name,index) in zip(quotes_descriptions_by_location.index, range(len(quotes_descriptions_by_location.index)))}

#%% Boolean Similarity ##########################################################
boolean_similarity = booleanSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("boolean_"+fileSuffix+".txt", boolean_similarity["df_similarity_boolean"] , fmt='%.3f', delimiter = " ")
boolean_similarity["df_similarity_boolean"].to_csv("boolean_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
boolean_similarity["topSimilar"].to_csv("boolean_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
boolean_similarity["bottomSimilar"].to_csv("boolean_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])
print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on locations with at least 10 quotes/description

freq_loc[freq_loc>10]
top_loc = freq_loc[freq_loc>10].index

df_similarity_boolean_top = boolean_similarity["df_similarity_boolean"].loc[top_loc,top_loc]

topSimilar = rankSimilarity(df_similarity_boolean_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_boolean_top, top = False, rank =10)

# save results
topSimilar.to_csv("boolean_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("boolean_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))


boolean_similarity_quote_description_by_loc = copy.deepcopy(boolean_similarity)

stacked_boolean_similarity_quote_description_by_loc  = stackDF(boolean_similarity_quote_description_by_loc ["df_similarity_boolean"],
                                              ["location1", "location2", "boolean_similarity"], symmetric = True)


stacked_boolean_similarity_quote_description_by_loc.to_csv("boolean_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)

#%% tf_idf ##################################################################

tfidf_similarity = tfidfSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("tfidf_"+fileSuffix+".txt",tfidf_similarity["df_similarity_tfidf"] , fmt='%.3f', delimiter = " ")
tfidf_similarity["df_similarity_tfidf"].to_csv("tfidf_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
tfidf_similarity["topSimilar"].to_csv("tfidf_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
tfidf_similarity["bottomSimilar"].to_csv("tfidf_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on locations with at least 10 quotes/description

freq_loc[freq_loc>10]
top_loc = freq_loc[freq_loc>10].index


df_similarity_tfidf_top = tfidf_similarity["df_similarity_tfidf"].loc[top_loc,top_loc]

topSimilar = rankSimilarity(df_similarity_tfidf_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_tfidf_top, top = False, rank =10)

# save results
topSimilar.to_csv("tf_idf_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("tf_idf_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

tfidf_similarity_quote_description_by_loc= copy.deepcopy(tfidf_similarity)

stacked_tfidf_similarity_quote_description_by_loc = stackDF(tfidf_similarity_quote_description_by_loc["df_similarity_tfidf"],
                                              ["location1", "location2", "tfidf_similarity"], symmetric = True)


stacked_tfidf_similarity_quote_description_by_loc.to_csv("tfidf_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)



#%% Analysis by document name = scene, documents = quotes+description #####

fileSuffix = "quotes_loc_by_scene"

# combine quotes-description (because want to remove duplicates
# that were generated when people speak at the same time)
combine =  df_data_lemmas["quote"] + "@" + df_data_lemmas["description"]
len(combine)
combine_unique = list(set(combine)) #1073

# alternative
# remove duplicates columns quote and description, keep first (default: Drop duplicates except for the first occurrence)
df_data_lemmas_unique = copy.deepcopy(df_data_lemmas)
df_data_lemmas_unique.drop_duplicates(['quote','description'], inplace=True) # Whether to drop duplicates in place or to return a copy


freq_scene = df_data_lemmas_unique.groupby('scene_name')['scene_name'].count().order(ascending = False)
print(freq_scene)

# save results
freq_scene.to_csv("freq_scene.csv", sep=",", header=True, index=True)

# this converts to dataframe with column1 = character, column2 = list of quotes
df_data_lemmas_unique.groupby('scene_name')['quote'].apply(lambda x: x.tolist()).iloc[0]

# this convert dataframe with column1 = character, column2 = list of tokens 
# (concatenates quotes with space in between then split to token)
# need to remove the last element of list because extra space as a result of concatenation
quotes_by_scene = df_data_lemmas_unique.groupby('scene_name')['quote'].apply(lambda x: (x + " ").sum().split(" ")[:-1])
quotes_by_scene = quotes_by_scene.map(lambda x: [token for token in x if token != '']) # remove "" tokens
type(quotes_by_scene) # series

descriptions_nodup = df_data_lemmas_unique.drop_duplicates(['description'])

# this converts to dataframe with column1 = character, column2 = list of quotes
descriptions_nodup.groupby('scene_name')['description'].apply(lambda x: x.tolist()).iloc[0]

# this convert dataframe with column1 = character, column2 = list of tokens 
# (concatenates quotes with space in between then split to token)
# need to remove the last element of list because extra space as a result of concatenation
descriptions_by_scene = descriptions_nodup.groupby('scene_name')['description'].apply(lambda x: (x + " ").sum().split(" ")[:-1])
descriptions_by_scene = descriptions_by_scene.map(lambda x: [token for token in x if token != '']) # remove "" tokens
type(descriptions_by_location) # series

# join series by location, column 1 = quotes, column2 = descriptions
quotes_descriptions_by_scene = pandas.concat([quotes_by_scene, descriptions_by_scene], axis=1, join='inner')
quotes_descriptions_by_scene.shape

# final data is a dataframe with index the location, column = quote_description as list of lemmas
quotes_descriptions_by_scene["quote_description"] = quotes_descriptions_by_scene["quote"] + quotes_descriptions_by_scene["description"]
quotes_descriptions_by_scene.drop('quote', axis=1, inplace=True)
quotes_descriptions_by_scene.drop('description', axis=1, inplace=True)
# need series for dictionary
type(quotes_descriptions_by_scene["quote_description"])


# key: document name (character), value = list of lemmas
doc_dic_normalized  = quotes_descriptions_by_scene["quote_description"].to_dict()

# key: name, value = index number
map_name_index = {name: index for (name,index) in zip(quotes_descriptions_by_scene.index, range(len(quotes_descriptions_by_scene.index)))}

#%% Boolean Similarity ##########################################################
boolean_similarity = booleanSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("boolean_"+fileSuffix+".txt", boolean_similarity["df_similarity_boolean"] , fmt='%.3f', delimiter = " ")
boolean_similarity["df_similarity_boolean"].to_csv("boolean_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
boolean_similarity["topSimilar"].to_csv("boolean_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
boolean_similarity["bottomSimilar"].to_csv("boolean_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(boolean_similarity["topSimilar"])
print("\n*Bottom*************************\n")

print(boolean_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

"""
# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on locations with at least 10 quotes/description

freq_loc[freq_loc>10]
top_loc = freq_loc[freq_loc>10].index

df_similarity_boolean_top = boolean_similarity["df_similarity_boolean"].loc[top_loc,top_loc]

topSimilar = rankSimilarity(df_similarity_boolean_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_boolean_top, top = False, rank =10)

# save results
topSimilar.to_csv("boolean_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("boolean_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('boolean_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
"""

boolean_similarity_quote_description_by_scene = copy.deepcopy(boolean_similarity)

stacked_boolean_similarity_quote_description_by_scene  = stackDF(boolean_similarity_quote_description_by_scene ["df_similarity_boolean"],
                                              ["scene1", "scene2", "boolean_similarity"], symmetric = True)


stacked_boolean_similarity_quote_description_by_scene.to_csv("boolean_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)

#%% tf_idf ##################################################################

tfidf_similarity = tfidfSimilarity(doc_dic_normalized, k = 10)

# save results
np.savetxt("tfidf_"+fileSuffix+".txt",tfidf_similarity["df_similarity_tfidf"] , fmt='%.3f', delimiter = " ")
tfidf_similarity["df_similarity_tfidf"].to_csv("tfidf_w_headers_"+fileSuffix+".csv", sep=",", header=True, index=True, float_format= '%.3f')    
        
tfidf_similarity["topSimilar"].to_csv("tfidf_top_"+fileSuffix+".csv", sep=",", header=True, index=True)
tfidf_similarity["bottomSimilar"].to_csv("tfidf_bottom_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(tfidf_similarity["topSimilar"])

print("\n*Bottom*************************\n")

print(tfidf_similarity["bottomSimilar"])

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

"""
# ignore documents with very few values (e.g. characters with only 1 or 2 quotes)
# as these will tend to be the most dissimlar
# focus on locations with at least 10 quotes/description

freq_loc[freq_loc>10]
top_loc = freq_loc[freq_loc>10].index


df_similarity_tfidf_top = tfidf_similarity["df_similarity_tfidf"].loc[top_loc,top_loc]

topSimilar = rankSimilarity(df_similarity_tfidf_top, top = True, rank =10) 
bottomSimilar = rankSimilarity(df_similarity_tfidf_top, top = False, rank =10)

# save results
topSimilar.to_csv("tf_idf_top2_"+fileSuffix+".csv", sep=",", header=True, index=True)
bottomSimilar.to_csv("tf_idf_bottom2_"+fileSuffix+".csv", sep=",", header=True, index=True)

# save output to output.txt
orig_stdout = sys.stdout
f = open('tf_idf_output2_'+fileSuffix+'.txt', 'w')
sys.stdout = f

print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))
                                       
f.close()
sys.stdout =  orig_stdout       


print("\n********************************\n")
print("\n*Top****************************\n")

print(topSimilar)

print("\n*Bottom*************************\n")

print(bottomSimilar)

print("\n********************************\n")
print("Output files saved in: {}\n".format(os.getcwd()))

"""

tfidf_similarity_quote_description_by_scene = copy.deepcopy(tfidf_similarity)

stacked_tfidf_similarity_quote_description_by_scene = stackDF(tfidf_similarity_quote_description_by_scene["df_similarity_tfidf"],
                                              ["scene1", "scene2", "tfidf_similarity"], symmetric = True)


stacked_tfidf_similarity_quote_description_by_scene.to_csv("tfidf_stacked_"+fileSuffix+".csv", sep=",", header=True, index=False)


##############################################################################
#%% Topic Modeling ######################################################

#%% Functions ###############################################################

#Attributes of model LDA

#components_	(array, shape = [n_topics, n_features]) Point estimate of the topic-word distributions (Phi in literature)
#topic_word_ :	Alias for components_
#nzw_	(array, shape = [n_topics, n_features]) Matrix of counts recording topic-word assignments in final iteration.
#ndz_	(array, shape = [n_samples, n_topics]) Matrix of counts recording document-topic assignments in final iteration.
#doc_topic_	(array, shape = [n_samples, n_features]) Point estimate of the document-topic distributions (Theta in literature)
#nz_	(array, shape = [n_topics]) Array of topic assignment counts in final iteration.

def getModelLDA(df,n_topics=100, n_iter=1500, random_state=2):
    """ Returns a model LDA object
    
    Args:
        df (pandas dataframe):
        n_topics (int): Number of topics (default 100)
        n_iter (int): Number of sampling iterations (default 1500)
        random_state (int):The generator used for the initial topics (default 2)
        
    Returns:
        model lDA object with the following attributes
        #components_	(array, shape = [n_topics, n_features]) Point estimate of the topic-word distributions (Phi in literature)
        #topic_word_ :	Alias for components_
        #nzw_	(array, shape = [n_topics, n_features]) Matrix of counts recording topic-word assignments in final iteration.
        #ndz_	(array, shape = [n_samples, n_topics]) Matrix of counts recording document-topic assignments in final iteration.
        #doc_topic_	(array, shape = [n_samples, n_features]) Point estimate of the document-topic distributions (Theta in literature)
        #nz_	(array, shape = [n_topics]) Array of topic assignment counts in final iteration.
        
    """
    
    tf_matrix = df.as_matrix()
    tf_matrix.shape # (79652, 249)
    
    # for lda, rows have to by documents, columns terms
    tf_matrix = np.transpose(tf_matrix)
    tf_matrix.shape
     
    tf_matrix.dtype  # dtype('float64')          
    tf_matrix = tf_matrix.astype(np.int64) # convert to integer
    tf_matrix.dtype 
    
    ####### Model #######
    model = lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_state)
    
    # before training/inference:
    #SOME_FIXED_SEED = 0124
    #np.random.seed(SOME_FIXED_SEED)
    
    t0 = time.time()
    model.fit(tf_matrix)
    
    t1 = time.time()
    print("Elapsed time:" + str(t1-t0)) # 27 minutes
    
    return model

def printLDA(df, model):
    """ Prints results from LDA
    
    Args:
        df (pandas dataframe): index = terms, columns = documents
        model (lda model)

    """

    docs = df.columns
    terms = df.index    
           
    n_top_words = 10
    
    ####### topic-word probabilities ######
    # a distribution over the N words in the vocabulary for each of the 20 
    # topics. For each topic, the probabilities of the words should be normalized.
    topic_word = model.topic_word_  # model.components_ also works
    
    # topics x terms
    print("type(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))
    
    # check normalized 
    for n in range(5):
        sum_pr = sum(topic_word[n,:])
        print("topic: {} sum: {}".format(n, sum_pr))
    
    ####### top 10 words for each topic (by probability) ####### 
    
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_words_decode = [t.encode('utf-8') for t in topic_words]
        print('Topic {}: {}'.format(i, ' '.join(topic_words_decode)))
        
    ####### document-topic probabilities ####### 
    # distribution over the 20 topics for each of the 395 documents. 
    # These should be normalized for each document
        
    # documents x topics
    doc_topic = model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic)))
    print("shape: {}".format(doc_topic.shape))
    
    # check normalized
    for n in range(5):
        sum_pr = sum(doc_topic[n,:])
        print("document: {} sum: {}".format(n, sum_pr))
    
    #######  sample the most probable topic ####### 
    for i in range(len(docs)):
        print("{} - {} (top topic: {})".format(i,docs[i].encode('utf-8'), doc_topic[i].argmax()))


def getTopicWords(df, model, n_top_words = 10):
    """ Gets the top words for each topic
    
    Args:
        df (pandas dataframe): index = terms, columns = documents
        model (lda model)
        n_top_words (int): top number of words for each topic (default = 10)
    
    Returns:
        pandas dataframe: index = topic, columns = counts, top words
        
    """
    
    docs = df.columns
    terms = df.index    
    
    # get counts of topic assignments for each topic
    model.nz_.shape # 100 topics
    
    # check counts match
    model.nzw_.sum(axis = 1).shape # sum across columns
    assert model.nz_.shape  == model.nzw_.sum(axis = 1).shape 
    
    topics = ["Topic " + str(i) for i in range(len(model.nz_))]
    
    counts = pandas.Series(model.nz_, index = topics)
    
    ####### topic-word probabilities ######
    # a distribution over the N words in the vocabulary for each of the 20 
    # topics. For each topic, the probabilities of the words should be normalized.
    topic_word = model.topic_word_  # model.components_ also works
    
    # get top 10 words for each topic (key = Topic i, value = string top 10 words)
    dicTop10w = {}
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(terms)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_words_decode = [t.encode('utf-8') for t in topic_words]
        top10str = ' '.join(topic_words_decode)
        dicTop10w[topics[i]] =  top10str 
        
    top10w = pandas.Series(dicTop10w)
    
    # create data frame: index: topic, columns: counts, top 10 words
    dfCounts = pandas.DataFrame( {"counts": counts, "top " + str(n_top_words) +" words": top10w} )
    
    # sort
    dfCounts_sorted = dfCounts.sort(["counts"], ascending = [False])
    dfCounts_sorted.index.name = "topic"
    
    return dfCounts_sorted 
     

def getDocTopics(df, model, n_top_topics = 10):
    """ Gets the top topics for each document
    
    Args:
        df (pandas dataframe): index = terms, columns = documents
        model (lda model)
        n_top_topics (int): top number of topics for each document (default = 10)
    
    Returns:
        pandas dataframe: index = document, topic columns = top1, top2, top3 ..topic 
    """
    
    docs = df.columns
    terms = df.index    
    
    topics = ["Topic " + str(i) for i in range(len(model.nz_))]
    topics_number = [i for i in range(len(model.nz_))]

    ####### document-topic probabilities ####### 
    # distribution over the 20 topics for each of the 395 documents. 
    # These should be normalized for each document
    
    # documents x topics
    doc_topic = model.doc_topic_
     
    # (key = document, value = dict (key: top1, top2.. value: topic number)
    dicTop10t = {}
    for i, doc_dist in enumerate(doc_topic):
        doc_topics = np.array(topics_number)[np.argsort(doc_dist)][:-(n_top_topics+1):-1]
        dicTop10t[docs[i]] = {"top" + str(j+1): doc_topics[j] for j in range(n_top_topics)}
        
    indices, column_values = zip(*dicTop10t.items())   
    df_doc_top10topics = pandas.DataFrame(list(column_values), index = list(indices),
                                          columns = ["top" + str(j+1) for j in range(n_top_topics)])
    
    df_doc_top10topics.index.name = "document"
    
    return df_doc_top10topics
    

def getDocTopicsDist(df, model):
    """ Get distribution of topics for each document
    
    Args:
        df (pandas dataframe): index = terms, columns = documents
        model (lda model)
    
    Returns:
        pandas dataframe: index = document, columns = topics distribution
    
    """
    # document x topic matrix. each row is distribution of topics for a document
    doc_topic = model.doc_topic_
    
    docs = df.columns
    terms = df.index 
    
    topics = ["Topic " + str(i) for i in range(len(model.nz_))]
    topics_number = [i for i in range(len(model.nz_))]
    
    df_doc_topics = pandas.DataFrame(doc_topic, index = docs, columns = topics)
    
    return df_doc_topics


def getTopicWordsDist(df, model):
    """ Get distribution of words for each topic
    
    Args:
        df (pandas dataframe): index = terms, columns = documents
        model (lda model)
    
    Returns:
        pandas dataframe: index = topics , columns =words distribtion
    
    """
    # topic x words matrix each row is distribution of words for a topic
    topic_word = model.topic_word_
    
    docs = df.columns
    terms = df.index 
    
    topics = ["Topic " + str(i) for i in range(len(model.nz_))]
    topics_number = [i for i in range(len(model.nz_))]
    
    df_topic_words = pandas.DataFrame(topic_word , index = topics, columns = terms)
    
    return df_topic_words 
    
def filterTopTopics(df, model, k =5):
    """ Filter top k topics overall (not for each document)
    
    Args:
        df (pandas Dataframe): example stacked df with topic column
        model (lda Model)
        k (int): top k (default =5)
        
    Returns:
        pandas DF: filtered df
    
    """
    topics = ["Topic " + str(i) for i in range(len(model.nz_))]
    counts = pandas.Series(model.nz_, index = topics)
    counts.sort(ascending = False)
    
    # Using the Python in operator on a Series tests for membership in the index, 
    # not membership among the values.
    # If this behavior is surprising, keep in mind that using in on a Python 
    # dictionary tests keys, not values, and Series are dict-like. To test for 
    # membership in the values, use the method isin():
    # For DataFrames, likewise, in applies to the column axis, testing for 
    # membership in the list of column name
    
    return df.loc[df["topic"].isin(counts[:k].index)]
    

#%%  lDA character ############################################################
dic_LDA = {} # key: character, location, scene, value: dictionary with lda outputs


fileSuffix = "quotes_by_char"
df = tfidf_similarity_quotes_by_char["term_document"]

stacked_cols1 =  ["topic", "character", "probabilty"]
stacked_cols2 =  ["word", "topic", "probabilty"]

model = getModelLDA(df,n_topics=30, n_iter=2000, random_state=1)
printLDA(df,model)

# get distribution of topics for each document
df_doc_topics = getDocTopicsDist(df, model)
stacked_df_doc_topics = stackDF(df_doc_topics, stacked_cols1 , symmetric = False)
stacked_df_doc_topics.to_csv("stacked_doc_topics_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# filter top k topics overall (not for each)
stacked_df_doc_topics_5 = filterTopTopics(stacked_df_doc_topics , model, k = 5)
stacked_df_doc_topics_5.to_csv("stacked_doc_topics_5_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get distribution of words for each document
df_topic_words = getTopicWordsDist(df, model)
stacked_df_topic_words = stackDF(df_topic_words, stacked_cols2 , symmetric = False)
stacked_df_topic_words.to_csv("stacked_topic_words_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get the top 10 words for each topic
df_topic_top10words = getTopicWords(df, model, n_top_words = 10)
# top10 = df_topic_top10words[:10]
df_topic_top10words.to_csv("topic_top10words_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

# get the top 10 topics for each document
df_doc_top10topics = getDocTopics(df, model, n_top_topics = 10)
df_doc_top10topics.to_csv("doc_top10topics_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

dic_LDA["character"]= {"model": copy.deepcopy(model),
                       "df_doc_topics": copy.deepcopy(df_doc_topics),
                       "stacked_df_doc_topics": copy.deepcopy(stacked_df_doc_topics),
                       "stacked_df_doc_topics_5": copy.deepcopy(stacked_df_doc_topics_5),
                       "df_topic_words": copy.deepcopy(df_topic_words),
                       "stacked_df_topic_words":  copy.deepcopy(stacked_df_topic_words),
                       "df_topic_top10words": copy.deepcopy(df_topic_top10words),
                       "df_doc_top10topics": copy.deepcopy(df_doc_top10topics)}
          

#%%  lDA location ############################################################
df = tfidf_similarity_quote_description_by_loc["term_document"]  
fileSuffix = "quotes_loc_by_loc"    

stacked_cols1 =  ["topic", "location", "probabilty"]
stacked_cols2 =  ["word", "topic", "probabilty"]

model = getModelLDA(df,n_topics=30, n_iter=2000, random_state=1)
printLDA(df,model)

# get distribution of topics for each document
df_doc_topics = getDocTopicsDist(df, model)
stacked_df_doc_topics = stackDF(df_doc_topics, stacked_cols1 , symmetric = False)
stacked_df_doc_topics.to_csv("stacked_doc_topics_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# filter top k topics overall (not for each)
stacked_df_doc_topics_5 = filterTopTopics(stacked_df_doc_topics , model, k = 5)
stacked_df_doc_topics_5.to_csv("stacked_doc_topics_5_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get distribution of words for each document
df_topic_words = getTopicWordsDist(df, model)
stacked_df_topic_words = stackDF(df_topic_words, stacked_cols2 , symmetric = False)
stacked_df_topic_words.to_csv("stacked_topic_words_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get the top 10 words for each topic
df_topic_top10words = getTopicWords(df, model, n_top_words = 10)
# top10 = df_topic_top10words[:10]
df_topic_top10words.to_csv("topic_top10words_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

# get the top 10 topics for each document
df_doc_top10topics = getDocTopics(df, model, n_top_topics = 10)
df_doc_top10topics.to_csv("doc_top10topics_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

dic_LDA["location"]= {"model": copy.deepcopy(model),
                       "df_doc_topics": copy.deepcopy(df_doc_topics),
                       "stacked_df_doc_topics": copy.deepcopy(stacked_df_doc_topics),
                       "stacked_df_doc_topics_5": copy.deepcopy(stacked_df_doc_topics_5),
                       "df_topic_words": copy.deepcopy(df_topic_words),
                       "stacked_df_topic_words":  copy.deepcopy(stacked_df_topic_words),
                       "df_topic_top10words": copy.deepcopy(df_topic_top10words),
                       "df_doc_top10topics": copy.deepcopy(df_doc_top10topics)}

#%%  lDA scene ############################################################            
df = tfidf_similarity_quote_description_by_scene["term_document"]  
fileSuffix = "quotes_loc_by_scene"
                     
stacked_cols1 =  ["topic", "scene", "probabilty"]
stacked_cols2 =  ["word", "topic", "probabilty"]

model = getModelLDA(df,n_topics=30, n_iter=2000, random_state=1)
printLDA(df,model)

# get distribution of topics for each document
df_doc_topics = getDocTopicsDist(df, model)
stacked_df_doc_topics = stackDF(df_doc_topics, stacked_cols1 , symmetric = False)
stacked_df_doc_topics.to_csv("stacked_doc_topics_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# filter top k topics overall (not for each)
stacked_df_doc_topics_5 = filterTopTopics(stacked_df_doc_topics , model, k = 5)
stacked_df_doc_topics_5.to_csv("stacked_doc_topics_5_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get distribution of words for each document
df_topic_words = getTopicWordsDist(df, model)
stacked_df_topic_words = stackDF(df_topic_words, stacked_cols2 , symmetric = False)
stacked_df_topic_words.to_csv("stacked_topic_words_" + fileSuffix +".csv", sep=",", header=True, index=False, encoding = 'utf-8')

# get the top 10 words for each topic
df_topic_top10words = getTopicWords(df, model, n_top_words = 10)
# top10 = df_topic_top10words[:10]
df_topic_top10words.to_csv("topic_top10words_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

# get the top 10 topics for each document
df_doc_top10topics = getDocTopics(df, model, n_top_topics = 10)
df_doc_top10topics.to_csv("doc_top10topics_" + fileSuffix +".csv", sep=",", header=True, index=True, encoding = 'utf-8')

dic_LDA["scene"]= {"model": copy.deepcopy(model),
                       "df_doc_topics": copy.deepcopy(df_doc_topics),
                       "stacked_df_doc_topics": copy.deepcopy(stacked_df_doc_topics),
                       "stacked_df_doc_topics_5": copy.deepcopy(stacked_df_doc_topics_5),
                       "df_topic_words": copy.deepcopy(df_topic_words),
                       "stacked_df_topic_words":  copy.deepcopy(stacked_df_topic_words),
                       "df_topic_top10words": copy.deepcopy(df_topic_top10words),
                       "df_doc_top10topics": copy.deepcopy(df_doc_top10topics)}


"""
#%% Plotting: Takes a long time  ##############################################
import matplotlib.pyplot as plt

# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass

# get index of top 5 topics
top5 = model.nz_.argsort()[-5:][::-1]
top5

t0 = time.time()

###### topic-word distribution ######
# The idea here is that each topic should have a distinct distribution of 
# words. In the stem plots below, the height of each stem reflects the
# probability of the word in the focus topic
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate(top5):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-50,4350)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()
plt.savefig('topic_word_distribution.png')
    
###### topic distribution for a few documents ######
# distributions give the probability of each of the 20 topics for every document
# many documents have more than one topic with high probability. As a result,
# choosing the topic with highest probability for each document can be subject to uncertainty
    
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([92, 139, 186, 197, 239]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 21)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()
plt.savefig('doc_topic_distribution.png')
    
t1 = time.time()
print(t1-t0) # 12 minutes
"""

##############################################################################
#%% Clustering ######################################################


#%% Functions #######################################################

def getHierarchicalClustering(tfidf_doc_terms, fileSuffix, method = "ward"):
    """ Saves picture of dendogram from HierarchicalClustering
    
    Args:
        tfidf_doc_terms (pandasdf): index = documents, columns = terms
        method (str): distance ward (default), complete or average
        fileSuffix (str): suffix to save file
           
    """
    doc_names = tfidf_doc_terms.index
    dist = 1 - cosine_similarity(tfidf_doc_terms)
    
    
    if method == 'complete':
        linkage_matrix = scipy.cluster.hierarchy.complete(dist) #define the linkage_matrix using ward clustering pre-computed distances
    
    elif method == "average":
        linkage_matrix = scipy.cluster.hierarchy.average(dist)
        
    else:
        linkage_matrix = scipy.cluster.hierarchy.ward(dist)
        
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels = doc_names);
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.tight_layout() #show plot with tight layout
    
    #uncomment below to save figure
    fileName = method +'_doc_clusters_' + fileSuffix + '.png'
    plt.savefig(fileName, dpi=300) #save figure as ward_clusters
    plt.close()

    print("Output file {} saved in: {}\n".format(fileName, os.getcwd()))


def getKMeansClustering(tfidf_doc_terms, fileSuffix, num_clusters = [3,4,5]):
    """ Runs k-means clustering for k in num_clusters
    Function also dumps km object in hard disk and saves pandas df with
    documents and cluster assignment as csv
    
    Args:
        tfidf_doc_terms (pandasdf): index = documents, columns = terms
        fileSuffix (str): suffix to save file
        num_clusters (list): k's (default =  [3,4,5])
        
    Returns:
        dict: key = number of cluster, value = pandas df with documents and cluster assignment
        
    """    
    
    dic_cluster = {}
    
    for k in num_clusters:
        km = KMeans(n_clusters=k)
        km.fit(tfidf_doc_terms)
        
        fileName = "kmeans" + str(k) +'_doc_clusters_' + fileSuffix + '.pkl'
        print("Output file {} saved in: {}\n".format(fileName, os.getcwd()))
    
        joblib.dump(km,fileName)
        # km = joblib.load('doc_cluster.pkl')
        
        
        # Get cluster assignment labels
        clusters = km.labels_.tolist()
    
        # Format results as a DataFrame
        results = pandas.DataFrame([tfidf_doc_terms.index, clusters]).T
        results.columns = ["document", "cluster"]
        
        fileName = "kmeans" + str(k) +'_doc_clusters_' + fileSuffix + '.csv'
        print("Output file {} saved in: {}\n".format(fileName, os.getcwd()))
        results.to_csv(fileName, sep=",", header=True, index=False)
    
        # number of documents per cluster
        results['cluster'].value_counts()
        
        dic_cluster[k] = results
        
    return dic_cluster
    
    


#%% By Characters #######################################################

fileSuffix = "quotes_by_char"
tfidf_doc_terms = tfidf_similarity_quotes_by_char["tf_idf"].transpose() # rows = documents (samples), columns = terms (features)
    
#### K-means
dic_cluster = getKMeansClustering(tfidf_doc_terms, fileSuffix, num_clusters = [3,4,5])

#### Hierarchical
for m in ["ward", "average", "complete"]:
    getHierarchicalClustering(tfidf_doc_terms, fileSuffix, method = m)

dic_cluster_quotes_by_char = copy.deepcopy(dic_cluster)

#%% By Location #######################################################

fileSuffix = "quotes_loc_by_loc"
tfidf_doc_terms = tfidf_similarity_quote_description_by_loc["tf_idf"].transpose() # rows = documents (samples), columns = terms (features)
    
#### K-means
dic_cluster = getKMeansClustering(tfidf_doc_terms, fileSuffix, num_clusters = [3,4,5])

#### Hierarchical
for m in ["ward", "average", "complete"]:
    getHierarchicalClustering(tfidf_doc_terms, fileSuffix, method = m)

dic_cluster_quote_description_by_location = copy.deepcopy(dic_cluster)

#%% By Scene #######################################################

fileSuffix = "quotes_loc_by_scene"
tfidf_doc_terms = tfidf_similarity_quote_description_by_scene["tf_idf"].transpose() # rows = documents (samples), columns = terms (features)
    
#### K-means
dic_cluster = getKMeansClustering(tfidf_doc_terms, fileSuffix, num_clusters = [3,4,5])

#### Hierarchical
for m in ["ward", "average", "complete"]:
    getHierarchicalClustering(tfidf_doc_terms, fileSuffix, method = m)

dic_cluster_quote_description_by_scene = copy.deepcopy(dic_cluster)

##############################################################################

#%% Dialogue Act Types ########################################################

# http://www.nltk.org/book/ch06.html
# When processing dialogue, it can be useful to think of utterances as a type
# of action performed by the speaker. This interpretation is most straightforward
# for performative statements such as "I forgive you" or "I bet you can't climb that
# hill." But greetings, questions, answers, assertions, and clarifications can all be
# thought of as types of speech-based actions. Recognizing the dialogue acts underlying
# the utterances in a dialogue can be an important first step in understanding the conversation.

# The NPS Chat Corpus, which was demonstrated in 1, consists of over 10,000 posts
# from instant messaging sessions. These posts have all been labeled with one of
# 15 dialogue act types, such as "Statement," "Emotion," "ynQuestion", and "Continuer."
# We can therefore use this data to build a classifier that can identify the dialogue
# act types for new instant messaging posts. The first step is to extract the basic
# messaging data. We will call xml_posts() to get a data structure representing the
# XML annotation for each post

#posts = nltk.corpus.nps_chat.xml_posts()[:10000]
posts = nltk.corpus.nps_chat.xml_posts()
len(posts) #10,567

#%%#### explore corpus

print(nltk.corpus.nps_chat.words())
print(nltk.corpus.nps_chat.tagged_words())
print(nltk.corpus.nps_chat.tagged_posts()) # doctest: +NORMALIZE_WHITESPACE
print(nltk.corpus.nps_chat.xml_posts()) # doctest: +ELLIPSIS
sorted(nltk.FreqDist(p.attrib['class'] for p in posts).keys())
posts[0].text
tokens = posts[0].findall('terminals/t')
[t.attrib['pos'] + "/" + t.attrib['word'] for t in tokens]

# define a simple feature extractor that checks what words the post contains
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set)) #67%
#nltk.metrics.precision()

classifier.classify_many([test_set[0][0],test_set[1][0]])
classifier.classify(test_set[0][0])
classifier.prob_classify_many([test_set[0][0],test_set[1][0]])
classifier.prob_classify(test_set[0][0])

for label in classifier.labels():
    print (label + ":" + str(classifier.prob_classify(test_set[0][0]).prob(label)))

classifier.labels()
classifier.show_most_informative_features(n=10)
classifier.most_informative_features(n=10)

#%%#### train and predict

classifier = nltk.NaiveBayesClassifier.train(featuresets)
classifier.labels()
classifier.show_most_informative_features(n=10)
classifier.most_informative_features(n=10)

# predict
features_to_predict = [dialogue_act_features(df_data_scenes["quote"][i]) for i in df_data_scenes.index]
predictions = classifier.classify_many(features_to_predict)

df_data_dialogue_act = copy.deepcopy(df_data_scenes)
df_data_dialogue_act["dialogue_act_type"] = predictions

df_data_dialogue_act.to_csv("df_data_dialogue_act.csv", sep=",", header=True, index=False)


##############################################################################

#%% Sentiment Analysis ########################################################

"""
#%% read data_clean.json file
with open("Code/new_data/data_names.json") as json_file:
    json_data = json.load(json_file)

# load json_data into pandas data frame
df_data = pandas.DataFrame(json_data, 
                           columns = ["time","character", "quote", "location", "description"]) 
                           
"""                           
#%% extract NER
df_data = copy.deepcopy(df_data_names)                           
# get 'quote' and 'description' columns 
quote_decrip = df_data[['quote', 'description']].values
quote_decrip = [str(item) for sublist in quote_decrip for item in sublist]


p2 = CoreNLP(configdict={'annotators':'tokenize, ssplit, pos, parse, lemma, ner'}, 
            #output_types=['pos','parse'],
            corenlp_jars=["/Users/Steven/Documents/corenlp/stanford-corenlp-full-2015-04-20/*"])
            
'''
creat a list to store (ner, lemma) tuples
Example:
[('PERSON', 'Riley'),
 ('PERSON', 'Riley'),
 ('PERSON', 'Riley')]
'''
lemma_ner = [] 
for line in quote_decrip:
    parsed = p2.parse_doc(line)["sentences"] 
    ner = [sentence["ner"] for sentence in parsed] #get ner by sentences
    ner = [str(item) for sublist in ner for item in sublist] #flat the list
    lemmas = [sentence["lemmas"] for sentence in parsed] 
    lemmas = [str(item) for sublist in lemmas for item in sublist]
    normner = [sentence["normner"] for sentence in parsed] 
    normner = [str(item) for sublist in normner for item in sublist]
    combo = [lemma_ner.append(x) for x in zip(ner,lemmas,normner) if x[0] != 'O' ] #filter out ner == 'O'

# create a dict to save and get frequency of each unique (ner,lemmas, normner) combo
ner_freq = dict()
for x in lemma_ner:
    if x not in ner_freq: # check if the combo exists
        ner_freq[x] = 1   # create a new (key, value) with count = 1
    else:
        ner_freq[x] = ner_freq[x] + 1 # adding counts

# create a dict to save Name Entities (PERSON, LOCATION, ORGANIZATION) and  Temporal References (DATE, TIME)
ner = dict()
for x in ner_freq.keys():
    freq = ner_freq[x]
    if x[0] not in ner.keys():
        if x[2] != '':
            ner[x[0]] = [(x[1],freq,x[2])]
        else:
            ner[x[0]] = [(x[1],freq)]
    else:
        if x[2] != '':
            ner[x[0]].append((x[1],freq,x[2]))
        else:
            ner[x[0]].append((x[1],freq))

#%% save ner dictionary into jason format file
with open('ner.json', 'w') as fp:
   json.dump(ner, fp, indent=2, encoding = 'utf-8')     
   
#%% Sentiment Analysis
"""   
# read lemmatized data
with open("Code/new_data/data_names.json") as json_file:
    json_data = json.load(json_file)

# load json_data into pandas data frame
lemma_data = pandas.DataFrame(json_data, 
                           columns = ["time","character", "quote", "location", "description"]) 
"""

lemma_data = copy.deepcopy(df_data_names)                           
# get pos, neg word lists (opinion_lexicon)
# source: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
pos_corpus = []
with open('opinion_lexicon/positive-words.txt') as inputfile:
    pos_corpus = inputfile.read() # read whole file as one string

pos_corpus = pos_corpus.split('\n') # split by new line


neg_corpus = []
with open('opinion_lexicon/negative-words.txt') as inputfile:
    neg_corpus = inputfile.read()

neg_corpus = neg_corpus.split('\n')

#%%
# creat new dictionary to save sentiment of each quote, key is index, value is sentiment
quote = dict()
for i in lemma_data.index:
    quote_lemma = lemma_data['quote'][i].split(' ') # split a quote and get each word
    sentiment_index = 0
    # check each word if in pos/neg corpus list and compute sentiment index
    for a in quote_lemma: 
        if str(a) in pos_corpus:
            #print('pos')
            sentiment_index += 1           
        if str(a) in neg_corpus:
            #print('neg')
            sentiment_index += -1
    '''       
    if sentiment_index > 0:
        sentiment = 1
    elif sentiment_index < 0:
        sentiment = -1
    else:
        sentiment = 0    
    '''
          
    quote[i] = sentiment_index

# convert quote dictionary to pandas dataframe
sentimentByIndex = pandas.DataFrame(quote.items(), columns=['Index', 'Sentiment'])
# append sentiment column back to overall data frame
lemma_data['Sentiment'] = pandas.Series(sentimentByIndex['Sentiment'], index=lemma_data.index)
# save data with sentiment by index to csv file
lemma_data.to_csv('sentiment_by_index2.csv')

#%% deduplicate repeated locations
quo_decri_time = copy.deepcopy(copy.deepcopy(lemma_data))
quo_decri_time.drop_duplicates(['quote','description'], inplace=True) # Whether to drop duplicates in place or to return a copy

quo_descri = dict()
for i in quo_decri_time.index:
    quote_descri = quo_decri_time['quote'][i].split(' ')
    quote_descri.append(quo_decri_time['description'][i].split(' '))# split a quote and get each word
    sentiment_index = 0
    # check each word if in pos/neg corpus list and compute sentiment index
    for a in quote_descri: 
        if str(a) in pos_corpus:
            #print('pos')
            sentiment_index += 1           
        if str(a) in neg_corpus:
            #print('neg')
            sentiment_index += -1
    
    quo_descri[i] = sentiment_index

# convert quote dictionary to pandas dataframe
index,sentiment = zip(*quo_descri.items())
d = {'sentiment' : pandas.Series(sentiment, index=index)}
sentimentBylocation = pandas.DataFrame(d)

quo_decri_time['Sentiment'] = pandas.Series(sentimentBylocation['sentiment'], index=quo_decri_time.index)
quo_decri_time.to_csv('sentiment_by_location.csv')

#%% sentiment by scenes
"""
with open("Code/new_data/data_scenes.json") as json_file:
    json_data = json.load(json_file)

# load json_data into pandas data frame
scene_data = pandas.DataFrame(json_data, 
                           columns = ["description","quote", "scene_number",
                                      "location", "time","scene_name","character"]) 
"""
scene_data = copy.deepcopy(df_data_scenes)

scene_data_uniq = copy.deepcopy(copy.deepcopy(scene_data))
scene_data_uniq.drop_duplicates(['quote','description'], inplace=True) # Whether to drop duplicates in place or to return a copy

quo_descri = dict()
for i in scene_data_uniq.index:
    quote_descri = scene_data_uniq['quote'][i].split(' ')
    quote_descri.append(scene_data_uniq['description'][i].split(' '))# split a quote and get each word
    sentiment_index = 0
    # check each word if in pos/neg corpus list and compute sentiment index
    for a in quote_descri: 
        if str(a) in pos_corpus:
            #print('pos')
            sentiment_index += 1           
        if str(a) in neg_corpus:
            #print('neg')
            sentiment_index += -1
    
    quo_descri[i] = sentiment_index

# convert quote dictionary to pandas dataframe
index,sentiment = zip(*quo_descri.items())
d = {'sentiment' : pandas.Series(sentiment, index=index)}
sentimentBylocation = pandas.DataFrame(d)

scene_data_uniq['Sentiment'] = pandas.Series(sentimentBylocation['sentiment'], index=quo_decri_time.index)
scene_data_uniq.to_csv('sentiment_by_scene.csv')
