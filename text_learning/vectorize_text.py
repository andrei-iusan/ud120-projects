#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""



def vectorize_emails(from_sara, from_chris, words_to_remove = [], max_emails = False):
    '''
    This function can be used independently. It takes as input the file pointers
    from_sara and from_chris, mandatory, and optionally:
    a list of words to remove (if not sent, it won't remove
            any words from the emails)
    and a max_emails (if parameter not sent, it uses all emails available,
            if max_emails is larger than the number of emails available,
            it uses all available emails)
    '''
    from_data = []
    word_data = []

    ### temp_counter is a way to speed up the development--there are
    ### thousands of emails from Sara and Chris, so running over all of them
    ### can take a long time
    ### temp_counter helps you only look at the first 200 emails in the list so you
    ### can iterate your modifications quicker
    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        temp_counter = 0
        for path in from_person:
            if max_emails and temp_counter == max_emails:
                break
            temp_counter += 1
            path = os.path.join('..', path[:-1])
            # print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email

            text = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            for word in words_to_remove:
                text = text.replace(word, '')
            ### append the text to word_data
            word_data.append(text)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            from_data.append(str(int(name == 'sara')))

            email.close()

    print "emails processed"
    from_sara.close()
    from_chris.close()

    return word_data, from_data

if __name__=='__main__':
    from_sara  = open("from_sara.txt", "r")
    from_chris = open("from_chris.txt", "r")
    words_to_remove = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]

    word_data, from_data = vectorize_emails(from_sara, from_chris, words_to_remove)
    pickle.dump( word_data, open("your_word_data.pkl", "w") )
    pickle.dump( from_data, open("your_email_authors.pkl", "w") )

    ## in Part 4, do TfIdf vectorization here
    # sw = stopwords.words("english")
    tfidfVec = TfidfVectorizer(stop_words = 'english')
    tfidfVec.fit(word_data)
