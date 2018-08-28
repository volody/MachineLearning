import os
import email
from email import policy
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import re
from html import unescape

SPAM_PATH = "../../input/spamclassifier/"

# LOAD DATA

# grab names
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

print('The ham  files count is {}.'.format(len(ham_filenames)))
print('The spam files count is {}.'.format(len(spam_filenames)))

# add loader
def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.message_from_binary_file(f, policy=policy.default)

# load them
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

# EXPLORE DATA

# check data
# print('------------- start of data ---------')
# print(ham_emails[1].get_content().strip())
# print('-------------  end  of data ---------')

# 
def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures        

# # print most_common ham / spam
# print(structures_counter(ham_emails).most_common())
# print(structures_counter(spam_emails).most_common())

# dum header in first spam email
# for header, value in spam_emails[0].items():
#     print(header,":",value)

# print(spam_emails[0]["Subject"])

# SPLIT DATA

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

# get html spam emails
html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]

# dump sample
sample_html_spam = html_spam_emails[7]
# print(sample_html_spam.get_content().strip()[:100], "...")    